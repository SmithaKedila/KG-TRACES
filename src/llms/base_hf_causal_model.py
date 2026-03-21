import os
import dotenv
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
)

from .base_language_model import BaseLanguageModel
from tools.logger_factory import setup_logger

logger = setup_logger("base_hf_causal_model")
dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


class HfCausalModel(BaseLanguageModel):
    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    @staticmethod
    def add_args(parser):
        parser.add_argument("--maximun_token", type=int, default=4096)
        parser.add_argument("--max_output_tokens", type=int, default=786)
        parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
        parser.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
        parser.add_argument(
            "--attn_implementation",
            default="flash_attention_2",
            choices=["eager", "sdpa", "flash_attention_2"],
        )
        parser.add_argument(
            "--generation_mode",
            type=str,
            default="greedy",
            choices=[
                "greedy",
                "beam",
                "sampling",
                "group-beam",
                "beam-early-stopping",
                "group-beam-early-stopping",
            ],
        )
        parser.add_argument("--k", type=int, default=1)
        parser.add_argument("--temperature", type=float, default=0.7)
        parser.add_argument("--top_p", type=float, default=0.85)
        parser.add_argument("--top_k", type=int, default=20)
        parser.add_argument("--repetition_penalty", type=float, default=1.05)
        parser.add_argument("--chat_model", default="true", type=lambda x: str(x).lower() == "true")

        parser.add_argument("--early_exit", action="store_true")
        parser.add_argument("--early_exit_threshold", type=float, default=None)
        parser.add_argument("--stop_string", type=str, default=None)
        parser.add_argument("--early_exit_min_tokens", type=int, default=8)

    def __init__(self, args):
        self.args = args
        self.maximun_token = args.maximun_token
        self.model_path = args.model_path

    def token_len(self, text):
        return len(self.tokenizer.tokenize(text))

    def prepare_for_inference(self):
        use_cuda = torch.cuda.is_available()

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if use_cuda:
            torch_dtype = torch.float16
            device_map = "auto"
            attn_impl = "sdpa"
        else:
            torch_dtype = torch.float32
            device_map = "cpu"
            attn_impl = "eager"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_impl,
        )

        gen_cfg = GenerationConfig.from_model_config(self.model.config)
        args = self.args
        gen_cfg.max_new_tokens = getattr(args, "max_output_tokens", 256)
        gen_cfg.temperature = getattr(args, "temperature", 0.7)
        gen_cfg.top_p = getattr(args, "top_p", 0.85)
        gen_cfg.top_k = getattr(args, "top_k", 20)
        gen_cfg.repetition_penalty = getattr(args, "repetition_penalty", 1.05)

        if getattr(args, "generation_mode", "greedy") == "greedy":
            gen_cfg.do_sample = False
        else:
            gen_cfg.do_sample = True

        self.generation_cfg = gen_cfg

    def prepare_model_prompt(self, query):
        if self.args.chat_model:
            chat_query = [{"role": "user", "content": query}]
            return self.tokenizer.apply_chat_template(
                chat_query,
                tokenize=False,
                add_generation_prompt=True,
            )
        return query

    @torch.inference_mode()
    def generate_sentence_batch(self, llm_input, *args, **kwargs) -> list[list[str]]:
        prompts = [self.prepare_model_prompt(x) for x in llm_input]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        input_ids = inputs.input_ids
        prompt_len = input_ids.shape[1]

        #logits_processor = LogitsProcessorList()
        #stopping_criteria = StoppingCriteriaList()


        generated_ids = self.model.generate(
            **inputs,
            generation_config=self.generation_cfg,
        )

        responses = []
        for batch_idx in range(input_ids.shape[0]):
            batch_responses = []
            for seq_idx in range(self.args.k):
                sequence = generated_ids[batch_idx * self.args.k + seq_idx]
                response_len = len(input_ids[batch_idx])
                decoded = "".join(
                    self.tokenizer.batch_decode(
                        sequence[response_len:],
                        skip_special_tokens=self.args.skip_special_tokens,
                    )
                )
                batch_responses.append(decoded.strip())
            responses.append(batch_responses)

        return responses

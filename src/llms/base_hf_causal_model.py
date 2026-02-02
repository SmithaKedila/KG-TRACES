import torch
from .base_language_model import BaseLanguageModel
import os
import dotenv

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig

from tools.logger_factory import setup_logger

logger = setup_logger("base_hf_causal_model")

dotenv.load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


class HfCausalModel(BaseLanguageModel):
    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    @staticmethod
    def add_args(parser):
        parser.add_argument("--maximun_token", type=int, help="max input token", default=4096)
        parser.add_argument("--max_output_tokens", type=int, help="max output token", default=786)

        parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
        parser.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
        parser.add_argument(
            "--attn_implementation",
            default="flash_attention_2",
            choices=["eager", "sdpa", "flash_attention_2"],
            help="enable flash attention 2",
        )
        parser.add_argument(
            "--generation_mode",
            type=str,
            default="greedy",
            choices=["greedy", "beam", "sampling", "group-beam", "beam-early-stopping", "group-beam-early-stopping"],
        )
        parser.add_argument(
            "--k", type=int, default=1, help="number of paths to generate"
        )
        parser.add_argument("--temperature", type=float, default=0.7)
        parser.add_argument("--top_p", type=float, default=0.85)
        parser.add_argument("--top_k", type=int, default=20)
        parser.add_argument("--repetition_penalty", type=float, default=1.05)
        parser.add_argument("--chat_model", default='true', type=lambda x: (str(x).lower() == 'true'))

    def __init__(self, args):
        self.args = args
        self.maximun_token = args.maximun_token
        self.model_path = args.model_path

    def token_len(self, text):
        return len(self.tokenizer.tokenize(text))

    def prepare_for_inference(self):
        import torch

        # 1) Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)

        # 2) Decide CPU vs GPU
        use_cuda = torch.cuda.is_available()

        if use_cuda:
            torch_dtype = torch.float16      # works well on your P100
            device_map = "auto"
            attn_impl = "sdpa"              # IMPORTANT: no flash_attention_2
        else:
            torch_dtype = torch.float32
            device_map = "cpu"
            attn_impl = "eager"

        # 3) Load model (from local path models/KG-TRACES)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_impl,
        )

        # 4) Create a GenerationConfig and store it as self.generation_cfg
        gen_cfg = GenerationConfig.from_model_config(self.model.config)

        # Override a few fields from args if they exist
        args = self.args
        gen_cfg.max_new_tokens = getattr(args, "max_output_tokens", 256)
        gen_cfg.temperature = getattr(args, "temperature", 0.7)
        gen_cfg.top_p = getattr(args, "top_p", 0.85)
        gen_cfg.top_k = getattr(args, "top_k", 20)
        gen_cfg.repetition_penalty = getattr(args, "repetition_penalty", 1.05)

        # Greedy vs sampling
        if getattr(args, "generation_mode", "greedy") == "greedy":
            gen_cfg.do_sample = False
        else:
            gen_cfg.do_sample = True

        self.generation_cfg = gen_cfg

    def prepare_model_prompt(self, query):
        if self.args.chat_model:
            chat_query = [
                {"role": "user", "content": query}
            ]
            return self.tokenizer.apply_chat_template(chat_query, tokenize=False, add_generation_prompt=True)
        else:
            return query
    
    @torch.inference_mode()
    def generate_sentence_batch(self, llm_input, *args, **kwargs) -> list[list[str]]:
        new_llm_input = []
        for input in llm_input:
            new_llm_input.append(self.prepare_model_prompt(input))

        if isinstance(new_llm_input, list):
            inputs = self.tokenizer(new_llm_input, return_tensors="pt", padding=True, truncation=True,).to(self.model.device)
        else:
            inputs = self.tokenizer([new_llm_input], return_tensors="pt",padding=True, truncation=True,).to(self.model.device)

        input_ids = inputs.input_ids

        generated_ids = self.model.generate(
            **inputs,
            generation_config=self.generation_cfg,
            )

        responses = []

        for batch_idx in range(input_ids.shape[0]):
            batch_responses = []
            for seq_idx in range(self.args.k):
                sequence = generated_ids[batch_idx *self.args.k + seq_idx] 
                response_len = len(input_ids[batch_idx])
                decoded = "".join(self.tokenizer.batch_decode(sequence[response_len:], skip_special_tokens=self.args.skip_special_tokens))
                batch_responses.append(decoded.strip()) 
            responses.append(batch_responses)

        return responses


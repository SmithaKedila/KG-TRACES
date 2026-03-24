import json
import sys
import os
import re
import math
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
import datasets
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    StoppingCriteriaList,
)

datasets.disable_progress_bar()
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

import utils
from tools.logger_factory import setup_logger

from qa_prediction.faithfulness import build_evidence_set, verify_triple_path

logger = setup_logger("gen_predict_path")

N_CPUS = int(os.environ["SLURM_CPUS_PER_TASK"]) if "SLURM_CPUS_PER_TASK" in os.environ else 1
PATH_RE = r"<PATH>(.*)<\/PATH>"

INSTRUCTION_RELATION = """Please generate a valid reasoning relation path that can be helpful for answering the following question:"""

#INSTRUCTION_TRIPLE = """Please generate a valid reasoning triple path that can be helpful for answering the following question
#- The reasoning triples path should follow the format: <PATH>subject<SEP>relation<SEP>object</PATH>.
#- If multiple triples are needed, output them in a logical sequence.
#- If no meaningful relation path can be generated, return <PATH>NONE</PATH>.
#"""
INSTRUCTION_TRIPLE = """Please generate a valid reasoning triple path that can be helpful for answering the following question
- The reasoning triples path should follow the format: <PATH>subject<SEP>relation<SEP>object</PATH>.
- If multiple triples are needed, output them in a logical sequence.
- If no meaningful relation path can be generated, return <PATH>NONE</PATH>.
- The reasoning triples path should follow the format:
  <PATH>subject -> relation -> object</PATH>
- If multiple triples are needed, output them in a logical sequence like:
  <PATH>s1 -> r1 -> o1 -> r2 -> o2</PATH>
- If no meaningful path can be generated, return <PATH>NONE</PATH>.
"""


def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        return open(path, "w"), []
    with open(path, "r") as f:
        processed_results = []
        for line in f:
            results = json.loads(line)
            processed_results.append(results["id"])
    return open(path, "a"), processed_results

def parse_relation_prediction(prediction):
    """
    Parse a list of predictions to a list of rules

    Args:
        prediction (_type_): _description_

    Returns:
        _type_: _description_
    """
    results = []
    for p in prediction:
        path = re.search(PATH_RE, p)
        if path is None:
            continue
        path = path.group(1)
        path = path.split("<SEP>")
        if len(path) == 0:
            continue
        rules = set()
        for rel in path:
            rel = rel.strip()
            if rel == "":
                continue
            rules.add(rel)
        results.append(list(rules))
    return results



def parse_triple_prediction(predictions):
    """
    Parse a list of predictions to a list of triples

    Args:
        prediction (_type_): _description_

    Returns:
        _type_: _description_
    """
    results = []
    for prediction in predictions:
        prediction = prediction.strip().split("\n")
        for p in prediction:
            path = re.search(PATH_RE, p)
            if not path:
                continue
            path = path.group(1)
            parts = path.split("->")
            if len(parts) % 2 != 1:
                continue
            triples = set()
            for i in range(0, len(parts)-2, 2):
                subject = parts[i].strip()
                relation = parts[i+1].strip()
                obj = parts[i+2].strip()
                triples.add((subject, relation, obj))
            results.append(list(triples))
    
    return results

def compute_triple_confidence(gen_ids, token_logprobs, tokenizer, triples):
    triple_confidences = []
    gen_tokens = gen_ids.tolist() if hasattr(gen_ids, "tolist") else list(gen_ids)

    for subj, rel, obj in triples:
        comp_confs = []

        for comp in [subj, rel, obj]:
            comp_tokens = tokenizer.encode(comp, add_special_tokens=False)
            L = len(comp_tokens)
            if L == 0:
                continue

            best_conf = None
            for i in range(len(gen_tokens) - L + 1):
                if gen_tokens[i : i + L] == comp_tokens:
                    span_lp = token_logprobs[i : i + L]
                    avg_lp = sum(span_lp) / len(span_lp)
                    conf = math.exp(avg_lp)
                    if best_conf is None or conf > best_conf:
                        best_conf = conf

            if best_conf is not None:
                comp_confs.append(best_conf)

        triple_confidences.append(sum(comp_confs) / len(comp_confs) if comp_confs else 0.0)

    return triple_confidences

def _top1_top2_gap(logits_1d: torch.Tensor) -> float:
    probs = torch.softmax(logits_1d.float(), dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    return float(top2[0] - top2[1])


def _mean_gap_confidence(step_logits, row_idx: int, gen_len: int) -> float:
    if gen_len <= 0:
        return 0.0
    T = min(gen_len, len(step_logits))
    if T <= 0:
        return 0.0
    s = 0.0
    for t in range(T):
        s += _top1_top2_gap(step_logits[t][row_idx])
    return s / T


def generate_seq(
    model,
    input_texts,
    tokenizer,
    num_beam=3,
    do_sample=False,
    max_new_tokens=100,
):
    inputs = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    batch_size = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_beams=num_beam,
        num_return_sequences=num_beam,
        early_stopping=True,
        do_sample=do_sample,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
        max_new_tokens=max_new_tokens,
    )

    all_sequences = output.sequences
    step_logits = output.logits

    predictions = tokenizer.batch_decode(
        all_sequences[:, prompt_len:],
        skip_special_tokens=True,
    )
    predictions = [p.strip() for p in predictions]
    predictions = [
        predictions[i * num_beam:(i + 1) * num_beam]
        for i in range(batch_size)
    ]

    if num_beam > 1:
        scores = output.sequences_scores.reshape(batch_size, num_beam).tolist()
        norm_scores = torch.softmax(
            output.sequences_scores.reshape(batch_size, num_beam), dim=-1
        ).tolist()
    else:
        scores = [[1.0] for _ in range(batch_size)]
        norm_scores = [[1.0] for _ in range(batch_size)]

    def entropy(p):
        return -sum(pi * math.log(pi + 1e-12) for pi in p)

    results = []
    for b in range(batch_size):
        beam_paths = predictions[b]
        beam_scores = scores[b]
        beam_norm = norm_scores[b]
        path_entropy = entropy(beam_norm)

        triple_confidences = []
        confidence_scores = []

        for beam_idx in range(num_beam):
            row_idx = b * num_beam + beam_idx
            seq = all_sequences[row_idx]
            gen_ids = seq[prompt_len:]
            gen_len = len(gen_ids)

            # latest confidence scoring: mean top1-top2 gap
            seq_conf = _mean_gap_confidence(
                step_logits=step_logits,
                row_idx=row_idx,
                gen_len=gen_len,
            )
            confidence_scores.append(seq_conf)

            parsed = parse_triple_prediction([beam_paths[beam_idx]])
            if len(parsed) == 0 or len(parsed[0]) == 0:
                triple_confidences.append([])
                continue

            token_logprobs = []
            for t, logits_t in enumerate(step_logits):
                if t >= gen_len:
                    break
                lp = torch.log_softmax(logits_t[row_idx], dim=-1)
                token_logprobs.append(lp[gen_ids[t]].item())

            triples = parsed[0]
            beam_triple_conf = compute_triple_confidence(
                gen_ids=gen_ids.tolist() if hasattr(gen_ids, "tolist") else gen_ids,
                token_logprobs=token_logprobs,
                tokenizer=tokenizer,
                triples=triples,
            )
            triple_confidences.append(beam_triple_conf)

        results.append({
            "paths": beam_paths,
            "scores": beam_scores,
            "path_conf": confidence_scores,   # keep latest confidence scoring
            "path_entropy": path_entropy,
            "triple_confidences": triple_confidences,
        })

    return results

def load_webqsp_split(split: str):
    if split not in ["train", "validation", "test"]:
        raise ValueError(f"Unsupported split: {split}")
    base_dir = os.path.join("data", "processed/webqsp")
    file_map = {
        "train": "train-triple_path.jsonl",
        "validation": "validation.jsonl",
        "test": "test.jsonl",
    }
    return load_dataset("json", data_files=os.path.join(base_dir, file_map[split]), split="train")


def load_cwq_split(split: str):
    if split not in ["train", "validation", "test"]:
        raise ValueError(f"Unsupported split: {split}")
    base_dir = os.path.join("data", "processed/cwq")
    file_map = {
        "train": "train-triple_path.jsonl",
        "validation": "validation.jsonl",
        "test": "test.jsonl",
    }
    return load_dataset("json", data_files=os.path.join(base_dir, file_map[split]), split="train")


def gen_prediction(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    output_dir = os.path.join(
        args.output_path,
        args.dataset,
        args.split,
        args.model_name,
        f"type_{args.path_type}",
    )
    os.makedirs(output_dir, exist_ok=True)

    dataset = load_webqsp_split(args.split) if args.dataset == "webqsp" else load_cwq_split(args.split)
    prompter = utils.InstructFormater(args.prompt_path)

    def prepare_dataset(sample):
        question = sample["question"] if sample["question"].endswith("?") else sample["question"] + "?"
        if args.path_type == "relation":
            sample["text"] = prompter.format(system=INSTRUCTION_RELATION, query="**Question**:\n" + question)
        else:
            sample["text"] = prompter.format(system=INSTRUCTION_TRIPLE, query="**Question**:\n" + question)
        return sample

    dataset = dataset.map(prepare_dataset, num_proc=N_CPUS)

    prediction_file = os.path.join(
        output_dir,
        f"predictions_{args.n_beam}_{args.do_sample}.jsonl",
    )
    f, processed_results = get_output_file(prediction_file, force=args.force)
    filter_data = [data for data in dataset if data["id"] not in processed_results]

    sample_latencies = []

    for batch_idx in tqdm(range(0, len(filter_data), args.batch_size), desc="Generating predict paths"):
        batch = filter_data[batch_idx : batch_idx + args.batch_size]
        batch_inputs = [d["text"] for d in batch]

        batch_start = time.time()
        raw_outputs = generate_seq(
            model=model,
            input_texts=batch_inputs,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            num_beam=args.n_beam,
            do_sample=args.do_sample,
        )
        batch_latency = time.time() - batch_start
        per_example_latency = batch_latency / max(len(batch_inputs), 1)

        for i in range(len(batch_inputs)):
            ro = raw_outputs[i]
            paths = ro["paths"]

            if args.path_type == "relation":
                parsed_paths = parse_relation_prediction(paths)
            else:
                parsed_paths = parse_triple_prediction(paths)

            sample_latencies.append(per_example_latency)

            output_data = {
                "id": batch[i]["id"],
                "question": batch[i]["question"],
                "answer": batch[i]["answer"],
                "prediction_paths": parsed_paths,
                "ground_paths": batch[i]["ground_paths"],
                "input": batch[i]["text"],
                "path_conf": ro["path_conf"],
                "path_entropy": ro["path_entropy"],
                "triple_confidences": ro["triple_confidences"],
                "latency_sec": per_example_latency,
            }
            # ✅ Added: Faithfulness verification for triple paths against sample["graph"]
            # - Adds:
            #     output_data["path_meta"] -> per-path verified + unsupported steps
            #     output_data["verified_paths"] -> only verified paths
            # - If --drop_unverified is enabled, prediction_paths becomes verified_paths
            if args.verify_paths and args.path_type == "triple":
                evidence = build_evidence_set(batch[i])  # uses batch[i]["graph"]
                verified_paths = []
                path_meta = []

                for path in parsed_paths:
                    # path is list of (subject, relation, object)
                    ok, bad = verify_triple_path(path, evidence)
                    path_meta.append({"verified": ok, "unsupported_steps": bad})
                    if ok:
                        verified_paths.append(path)

                output_data["path_meta"] = path_meta
                output_data["verified_paths"] = verified_paths

                if args.drop_unverified:
                    output_data["prediction_paths"] = verified_paths
            f.write(json.dumps(output_data) + "\n")
            f.flush()

    lat = np.array(sample_latencies) if sample_latencies else np.array([0.0])
    hist, bins = np.histogram(lat, bins=20)
    with open("latency_histogram.json", "w") as h:
        json.dump({"bins": bins.tolist(), "counts": hist.tolist()}, h, indent=2)

    f.close()
    return prediction_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--dataset", "-dataset", type=str, default="webqsp")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--path_type", type=str, default="relation", help="relation or triple")
    parser.add_argument("--output_path", type=str, default="results/gen_predict_path")
    parser.add_argument("--model_name", type=str, default="KG-TRACES")
    parser.add_argument("--model_path", type=str, default="model/KG-TRACES")
    parser.add_argument("--prompt_path", type=str, default="prompts/qwen2.5.txt")
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--n_beam", type=int, default=3)
    parser.add_argument("--do_sample", action="store_true")
    

    parser.add_argument(
        "--verify_paths",
        action="store_true",
        help="Verify predicted paths against sample['graph'] before saving.",
    )
    parser.add_argument(
        "--drop_unverified",
        action="store_true",
        help="If enabled, keep only verified paths in 'prediction_paths'.",
    )

    parser.add_argument("--stop_string", type=str, default="</PATH>", help="relation or triple")
    parser.add_argument("--early_exit_threshold", type=float, default=None)
    parser.add_argument("--early_exit_min_tokens", type=int, default=8)
    parser.add_argument("--early_exit_min_sep_count", type=int, default=2)

    args = parser.parse_args()
    logger.info(args)
    gen_prediction(args)

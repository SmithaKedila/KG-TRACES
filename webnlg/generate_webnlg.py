#!/usr/bin/env python3
"""
generate_webnlg.py

Usage:
  python webnlg/generate_webnlg.py \
    --model google/flan-t5-base \
    --max_examples 100 \
    --batch_size 8 \
    --out outputs/webnlg/webnlg_predictions.jsonl

Notes:
 - This script will try to load model/tokenizer from local cache first
   (useful when HPC has no internet). If not found and env allows, it
   will fall back to online download.
 - Output: newline JSON (jsonl) with {"prediction":..., "reference":...}
 - Metrics summary saved to same directory as JSONL with suffix _metrics_summary.json
"""
import argparse
import json
import os
import sys
from time import time
from typing import List

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from datasets import load_dataset
    import evaluate
except Exception as e:
    print("Missing module. Make sure your conda env has transformers, datasets, evaluate, torch installed.")
    raise

def get_model_and_tokenizer(model_name: str, local_only_env: bool):
    """
    Try loading from local cache first; if fails and local_only_env is False, try online.
    Returns tokenizer, model, device
    """
    local_only = True  # first try local-only
    # respect environment variables if offline is explicitly set
    if local_only_env:
        local_only = True
    for attempt_local in (True, False):
        if attempt_local and not local_only:
            # if local_only_env is False but we are on first attempt_local True, still allow local_only True
            pass
        try:
            # transformers uses local_files_only kw
            kwargs = {"local_files_only": attempt_local}
            tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            return tokenizer, model, device
        except Exception as exc:
            # if local-only was enforced or this attempt failed, move on and retry (or finally raise)
            if attempt_local:
                # try next: online (if allowed)
                if local_only_env:
                    # offline required -> raise
                    raise RuntimeError(
                        f"Failed to load model/tokenizer locally for '{model_name}' while offline is required. "
                        "Make sure model files are cached on HPC."
                    ) from exc
                else:
                    # fallthrough to next attempt (online)
                    print(f"[info] local cache load failed ({exc}). Will try online download next (if allowed).")
                    continue
            else:
                # attempt_local == False (online attempt) failed -> raise
                raise
    raise RuntimeError("Unable to load model/tokenizer")

def triples_to_input(triples: List[str]):
    """Converts a list of triple strings to a single input prompt string."""
    # join triples with ' ; ' which is readable and compact
    # you can customize the prompt template here
    joined = " ; ".join([t.strip() for t in triples if t and t.strip()])
    prompt = f"Generate a concise English sentence(s) from these RDF triples: {joined}"
    return prompt

def generate_batch(model, tokenizer, device, inputs: List[str], max_new_tokens=128, batch_device_size=8):
    """
    Generate outputs for a list of input strings.
    Returns list of decoded strings.
    """
    # encode
    enc = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)
    # generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
            early_stopping=True,
        )
    preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/flan-t5-base", help="HF model name or local path")
    ap.add_argument("--max_examples", type=int, default=None, help="Max examples from test set (None => all)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", type=str, default="webnlg_predictions.jsonl")
    ap.add_argument("--local_offline", action="store_true",
                    help="If set, force local_files_only for transformers (HPC offline mode).")
    args = ap.parse_args()

    model_name = args.model
    max_examples = args.max_examples
    batch_size = args.batch_size
    out_path = args.out

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    metrics_path = out_path.replace(".jsonl", "_metrics_summary.json")

    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
    local_offline_env = bool(os.environ.get("TRANSFORMERS_OFFLINE") or os.environ.get("HF_DATASETS_OFFLINE")) or args.local_offline

    # load model and tokenizer (local-first)
    try:
        tokenizer, model, device = get_model_and_tokenizer(model_name, local_only_env=local_offline_env)
    except Exception as e:
        print("Error loading model/tokenizer:", e)
        sys.exit(2)

    # load dataset (use web_nlg release_v3.0_en)
    ds = load_dataset("web_nlg", "release_v3.0_en", trust_remote_code=True)
    split = "test"
    data = ds[split]
    total = len(data)
    print("Total test examples:", total)

    if max_examples is not None and max_examples > 0:
        total = min(total, max_examples)

    # metrics
    bleu = evaluate.load("bleu")
    chrf = evaluate.load("chrf")
    rouge = evaluate.load("rouge")

    # iter and generate
    jsonl_out = open(out_path, "w", encoding="utf-8")
    all_preds = []
    all_refs = []
    start_time = time()

    print("Generating summaries...")
    processed = 0
    for i in range(0, total, batch_size):
        batch_indices = list(range(i, min(i + batch_size, total)))
        inputs = []
        refs = []
        for idx in batch_indices:
            example = data[idx]
            # example["original_triple_sets"] is a dict {'otriple_set': [[triple1, triple2,...], ...]}
            # choose first triple set (many examples contain list-of-sets)
            try:
                otriple_sets = example.get("original_triple_sets", {})
                # select first available triple list
                triple_lists = list(otriple_sets.get("otriple_set", []))
                if len(triple_lists) == 0:
                    # fallback to modified_triple_sets
                    triple_lists = list(example.get("modified_triple_sets", {}).get("mtriple_set", []))
                triples = triple_lists[0] if len(triple_lists) > 0 else []
            except Exception:
                triples = []

            input_prompt = triples_to_input(triples)
            # pick reference human text: the lex[text] field contains target sentences
            ref_texts = []
            try:
                lex = example.get("lex", {})
                text_list = lex.get("text", [])
                if isinstance(text_list, list) and len(text_list) > 0:
                    ref_texts.append(text_list[0])
            except Exception:
                pass
            # if no reference, fallback to empty string
            ref = ref_texts[0] if ref_texts else ""

            inputs.append(input_prompt)
            refs.append(ref)

        # generate for this batch
        preds = generate_batch(model, tokenizer, device, inputs, max_new_tokens=128)
        for p, r in zip(preds, refs):
            record = {"prediction": p.strip(), "reference": r.strip()}
            jsonl_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            all_preds.append(p.strip())
            all_refs.append(r.strip())
        processed += len(preds)
        if processed % 50 == 0 or processed == total:
            print(f"Processed {processed}/{total}")

    jsonl_out.close()
    elapsed = time() - start_time

    # compute metrics
    print("\nComputing metrics on generated samples...")
    # BLEU expects tokenized references and predictions (list of references lists)
    try:
        bleu_res = bleu.compute(predictions=all_preds, references=[[r] for r in all_refs])
    except Exception:
        # fallback to simple compute via evaluate with single references
        bleu_res = bleu.compute(predictions=all_preds, references=all_refs)
    chrf_res = chrf.compute(predictions=all_preds, references=all_refs)
    rouge_res = rouge.compute(predictions=all_preds, references=all_refs)

    metrics_summary = {
        "BLEU": float(bleu_res.get("bleu", bleu_res.get("score", 0.0))),
        "chrF": float(chrf_res.get("score", 0.0)),
        "ROUGE": rouge_res,
        "processed": len(all_preds),
        "time_minutes": elapsed / 60.0,
    }

    # write metrics
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics_summary, fh, indent=2, ensure_ascii=False)

    print("\nBLEU:", metrics_summary["BLEU"])
    print("chrF:", metrics_summary["chrF"])
    print("ROUGE:", metrics_summary["ROUGE"])
    print("\nTime taken: {:.2f} minutes".format(metrics_summary["time_minutes"]))
    print(f"\nSaved standardized predictions to {out_path}")
    print(f"Saved metrics summary to {metrics_path}")

if __name__ == "__main__":
    main()

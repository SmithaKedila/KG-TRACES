import argparse
import json
import os
import time

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF
from rouge_score import rouge_scorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/flan-t5-base")
    parser.add_argument("--max_examples", type=int, default=100)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Model:", args.model)

    # Load model from local cache (offline safe)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, local_files_only=True)
    model.to(device)
    model.eval()

    dataset = load_dataset(
        "web_nlg",
        "release_v3.0_en",
        split="test",
        trust_remote_code=True
    )

    total_examples = len(dataset)
    print("Total test examples:", total_examples)

    max_examples = args.max_examples if args.max_examples > 0 else total_examples
    max_examples = min(max_examples, total_examples)
    print("Running examples:", max_examples)

    bleu_metric = BLEU()
    chrf_metric = CHRF()
    rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    predictions = []
    references = []

    start_time = time.time()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w") as f:
        for i in range(max_examples):
            example = dataset[i]

            triples_list = example["original_triple_sets"]["otriple_set"][0]
            input_text = " ; ".join(triples_list)

            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    num_beams=4,
                    early_stopping=True
                )

            pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            ref = example["lex"]["text"][0].strip()

            predictions.append(pred)
            references.append(ref)

            # ---- Per-example metrics ----
            bleu_score_single = bleu_metric.sentence_score(pred, [ref]).score
            chrf_score_single = chrf_metric.sentence_score(pred, [ref]).score
            rouge_scores = rouge.score(ref, pred)

            example_output = {
                "prediction": pred,
                "reference": ref,
                "bleu": bleu_score_single,
                "chrf": chrf_score_single,
                "rouge1": rouge_scores["rouge1"].fmeasure,
                "rouge2": rouge_scores["rouge2"].fmeasure,
                "rougeL": rouge_scores["rougeL"].fmeasure
            }

            json.dump(example_output, f)
            f.write("\n")

            if i % 50 == 0:
                print(f"Processed {i+1}/{max_examples}")

    # ---- Corpus-level metrics ----
    bleu_score = bleu_metric.corpus_score(predictions, [references]).score
    chrf_score = chrf_metric.corpus_score(predictions, [references]).score

    rouge1_list, rouge2_list, rougeL_list = [], [], []
    for p, r in zip(predictions, references):
        scores = rouge.score(r, p)
        rouge1_list.append(scores["rouge1"].fmeasure)
        rouge2_list.append(scores["rouge2"].fmeasure)
        rougeL_list.append(scores["rougeL"].fmeasure)

    rouge_results = {
        "rouge1": sum(rouge1_list) / len(rouge1_list),
        "rouge2": sum(rouge2_list) / len(rouge2_list),
        "rougeL": sum(rougeL_list) / len(rougeL_list),
    }

    elapsed_minutes = (time.time() - start_time) / 60.0

    print("\nCorpus BLEU:", bleu_score)
    print("Corpus chrF:", chrf_score)
    print("Corpus ROUGE:", rouge_results)
    print("\nTime taken: {:.2f} minutes".format(elapsed_minutes))

    metrics_path = args.out.replace(".jsonl", "_metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "BLEU": bleu_score,
            "chrF": chrf_score,
            "ROUGE": rouge_results,
            "Time_minutes": elapsed_minutes
        }, f, indent=2)

    print(f"\nSaved standardized predictions to {args.out}")
    print(f"Saved metrics summary to {metrics_path}")


if __name__ == "__main__":
    main()

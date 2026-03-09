#!/bin/bash

MODEL="google/flan-t5-base"
MAX_EXAMPLES=50

mkdir -p outputs/webnlg

python webnlg/generate_webnlg.py \
  --model "$MODEL" \
  --max_examples "$MAX_EXAMPLES" \
  --out outputs/webnlg/webnlg_predictions.jsonl


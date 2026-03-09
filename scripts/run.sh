#!/bin/bash
set -e

TASK="$1"

if [ -z "$TASK" ]; then
    echo "Usage: bash scripts/run.sh [kgtraces | webnlg]"
    exit 1
fi

if [ "$TASK" == "kgtraces" ]; then
    echo "=== Submitting KG-TRACES jobs ==="
    sbatch kg-traces-gen-pred-path.sbatch
    sbatch kg-traces.sbatch

elif [ "$TASK" == "webnlg" ]; then
    echo "=== Submitting WebNLG job ==="
    sbatch webnlg.sbatch

else
    echo "Invalid option."
    echo "Usage: bash scripts/run.sh [kgtraces | webnlg]"
fi

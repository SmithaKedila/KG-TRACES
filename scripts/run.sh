#!/bin/bash
set -e

TASK="$1"

if [ "$TASK" == "kgtraces" ]; then
    sbatch kg-traces-gen-pred-path.sbatch
    sbatch kg-traces.sbatch
elif [ "$TASK" == "webnlg" ]; then
    sbatch webnlg.sbatch
else
    echo "Usage: bash scripts/run.sh kgtraces | webnlg"
fi

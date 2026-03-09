#!/bin/bash
#SBATCH --job-name=KG-traces-gen
#SBATCH --partition=nsfqs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

echo "=== Initializing Conda ==="

source /app/rl9.x/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /home/018178407/.conda/envs/webnlg

echo "Active Python:"
which python

python -c "import torch, datasets, transformers; print('ENV OK')"

cd /home/018178407/projects/KG-TRACES || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== Running KG-TRACES Path Generation ==="

bash src/qa_prediction/predict_path.sh

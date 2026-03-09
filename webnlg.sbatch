#!/bin/bash
#SBATCH --job-name=webnlg-run
#SBATCH --partition=nsfqs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

echo "=== Initializing Conda ==="

# Use system Anaconda base explicitly
source /app/rl9.x/anaconda3/2025.06/etc/profile.d/conda.sh

# Activate exact environment by full path
conda activate /home/018178407/.conda/envs/webnlg

echo "Active Python:"
which python

python -c "import torch; print('Torch OK')"

cd /home/018178407/projects/KG-TRACES || exit 1

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "=== Running WebNLG ==="

bash scripts/predict_webnlg.sh

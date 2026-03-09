#!/bin/bash
#SBATCH --job-name=webnlg-run
#SBATCH --partition=nsfqs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module load anaconda/2025.06
source $(conda info --base)/etc/profile.d/conda.sh
conda activate webnlg

cd $HOME/projects/KG-TRACES || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

bash scripts/predict_webnlg.sh

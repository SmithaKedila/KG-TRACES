#!/bin/bash
#SBATCH --job-name=KG-traces-run
#SBATCH --partition=nsfqs          # or gpuqm/gpuql
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00            # might need gpuqm/gpuql depending on limits
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

echo "=== Activating environment ==="
source ~/anaconda3/bin/activate kg_traces

echo "Running on host: $(hostname)"
echo "CUDA devices visible: $CUDA_VISIBLE_DEVICES"

echo "=== GPU CHECK ==="
nvidia-smi || echo "nvidia-smi not found or no GPU visible"

python - << 'PY'
import torch
print("Torch CUDA available:", torch.cuda.is_available())
print("Torch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
PY

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=== Running KG-TRACES prediction ==="
cd /home/018228028/KG-TRACES

bash scripts/predict.sh

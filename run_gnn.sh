#!/bin/bash
#SBATCH --job-name=gnn-brain
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=48g
#SBATCH -e slurm-gnn-%j.err
#SBATCH -o slurm-gnn-%j.out
#SBATCH --partition=athena
#SBATCH --account=pl217
#SBATCH --gres=gpu:1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate connecto

# Set environment variables to maximize performance
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export VECLIB_MAXIMUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

# PyTorch specific optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# Set PyTorch Geometric environment variables
export TORCH_GEOMETRIC_ALLOW_COPY_TENSOR=1

# Run GNN training script
echo "Starting GNN training with 16 CPUs and 1 GPU at $(date)"
python scripts/run_gnn_training.py --config configs/gnn_config.yaml

echo "GNN Training completed at $(date)"

# Cleanup
conda deactivate 
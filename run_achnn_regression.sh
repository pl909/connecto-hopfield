#!/bin/bash
#SBATCH --job-name=achnn-regression  # Changed job name
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=48g
#SBATCH -e slurm-achnn-regression-%j.err   # Changed error file name
#SBATCH -o slurm-achnn-regression-%j.out   # Changed output file name
#SBATCH --partition=athena
#SBATCH --account=pl217
#SBATCH --gres=gpu:1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate connecto

# Set environment variables to maximize performance (mostly relevant for numpy/MKL operations)
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export VECLIB_MAXIMUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

# PyTorch specific optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# Run ACHNN regression training script
echo "Starting ACHNN regression training with 16 CPUs and 1 GPU at $(date)"
python scripts/run_achnn_regression.py  # Changed script and config

echo "ACHNN Regression Training completed at $(date)"

# Cleanup
conda deactivate 
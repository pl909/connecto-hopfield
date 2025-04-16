#!/bin/bash
#SBATCH --job-name=fc-mlp-abide  # Changed job name
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1        # FC-MLP might not need a GPU, but keep for now if needed
#SBATCH --mem=48g                # Adjust memory if FC calculation or MLP needs more/less
#SBATCH -e slurm-fc-mlp-%j.err   # Changed error file name
#SBATCH -o slurm-fc-mlp-%j.out   # Changed output file name
#SBATCH --partition=athena
#SBATCH --account=pl217
#SBATCH --gres=gpu:1             # Requesting GPU

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate connecto

# Set environment variables to maximize performance (mostly relevant for numpy/MKL in FC calculation)
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export VECLIB_MAXIMUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

# PyTorch specific optimizations (less critical for MLP, but harmless)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# Run FC-MLP training script
echo "Starting FC-MLP training with 16 CPUs and 1 GPU at $(date)"
python scripts/run_fc_mlp_training.py configs/fc_mlp_config.yaml # Changed script and config

echo "FC-MLP Training completed at $(date)"

# Cleanup
conda deactivate
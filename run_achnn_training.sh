#!/usr/bin/bash
#SBATCH --job-name=connecto-hopfield
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=48g
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --partition=athena
#SBATCH --account=pl217
#SBATCH --gres=gpu:1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate connecto

# Set environment variables to maximize performance
export OMP_NUM_THREADS=16                 # Use all 16 CPUs for OpenMP parallelism
export MKL_NUM_THREADS=16                 # Use all 16 CPUs for MKL operations
export NUMEXPR_NUM_THREADS=16             # Numexpr parallel threads
export VECLIB_MAXIMUM_THREADS=16          # Apple Accelerate framework threads
export OPENBLAS_NUM_THREADS=16            # OpenBLAS threads

# PyTorch specific optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 # Reduce memory fragmentation
export CUDA_LAUNCH_BLOCKING=0             # Asynchronous CUDA kernel launches
export TORCH_CUDNN_V8_API_ENABLED=1       # Enable cuDNN v8 API 

# Run with optimized settings
echo "Starting ACHNN training with 16 CPUs and 1 GPU at $(date)"
python scripts/run_achnn_training.py configs/achnn_config.yaml

echo "Training completed at $(date)"

# Cleanup
conda deactivate 
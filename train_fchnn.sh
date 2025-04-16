#!/bin/bash
#SBATCH --job-name=achnn-regression  # Changed job name
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=50g
#SBATCH -e slurm-fchnn%j.err   # Changed error file name
#SBATCH -o slurm-fchnn%j.out   # Changed output file name
#SBATCH --partition=athena
#SBATCH --account=pl217
#SBATCH --gres=gpu:1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate frankenstein

# Run ACHNN regression training script
echo "Starting ACHNN regression training with 16 CPUs and 1 GPU at $(date)"
python build_custom_fchnn_embedding.py

echo "Frankenstein Training completed at $(date)"

# Cleanup
conda deactivate 
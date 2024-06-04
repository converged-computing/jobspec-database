#!/bin/bash
#SBATCH --partition= # set partition name
#SBATCH -w  # set node name
#SBATCH --gpus= # set number of GPUs
#SBATCH --cpus-per-gpu= # set number of CPUs per GPU
#SBATCH --mem=1
#SBATCH -J multi-gpu-training
#SBATCH -o logs/multi-gpu-training_%A_%a.out
#SBATCH -e logs/multi-gpu-training_%A_%a.err
#SBATCH -A # set account name


singularity exec --pwd $(pwd) --nv \
  -B /myovision:/mnt \
  image \
  bash -c "cd /mnt/myovision-sam && python3 inference.py"

#!/bin/bash

## Configure job
#SBATCH -c 24
#SBATCH -w compute-1-5
#SBATCH -t 72:00:00
#SBATCH --gres=gpu:1
#SBATCH -p gpu_high
#SBATCH --mem=64g
module load singularity # this is for singularity
ulimit -n 40000 # this is for singularity and large memory jobs, you could change 40000 to suitable numbers

# Kill all GPU processes before training, since there might be some "ghost" still running
kill -9 $(nvidia-smi | sed -n 's/|\s*[0-9]*\s*\([0-9]*\)\s*.*/\1/p' | sort | uniq | sed '/^$/d')

CUDA_VISIBLE_DEVICES=0 python xxx.py

# if using singularity
# CUDA_VISIBLE_DEVICES=0 singularity exec --nv your_image.img python xxx.py

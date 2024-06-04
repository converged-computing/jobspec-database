#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Request 8 CPU cores
#SBATCH -n 8

#SBATCH -t 00:05:00
#SBATCH -o hw_05.out

# Load CUDA module
module load cuda/10.0.130

nvidia-smi
nvprof -f -o app.nvpf ./bin/app
# ./bin/app

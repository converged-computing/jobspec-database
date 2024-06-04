#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=00:30:00

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu --gres=gpu:1 --gres-flags=enforce-binding

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (4GB) (CPU RAM):
#SBATCH --mem=4G

# Specify a job name:
#SBATCH -J MyKerasJob

# Specify an output file
#SBATCH -o KerasJob.out
#SBATCH -e KerasJob.out


# Set up the environment by loading modules
module load keras/2.0.9
module load cuda/8.0.61 cudnn/5.1 tensorflow/1.1.0_gpu

# Run a script
python mnist_cnn.py


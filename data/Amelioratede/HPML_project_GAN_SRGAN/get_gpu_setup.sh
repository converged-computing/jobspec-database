#!/bin/bash
#SBATCH --job-name=setup
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=16GB
#SBATCH --time=14:00:00
#SBATCH --output=setup.txt
#SBATCH --gres=gpu:rtx8000:4

nvidia-smi -L
nvidia-smi -l 60

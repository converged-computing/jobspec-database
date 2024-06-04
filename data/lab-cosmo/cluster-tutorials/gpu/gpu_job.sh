#!/bin/bash

#SBATCH --job-name=gpu-info
#SBATCH --get-user-env
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:05:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module purge
module load hpc-sdk

# YOUR GPU DEPENDING CODE
nvidia-smi > gpu.info

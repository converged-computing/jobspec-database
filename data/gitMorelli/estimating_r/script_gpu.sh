#!/bin/bash
#SBATCH --job-name=amorelli_job
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --ntasks=1
#SBATCH --mem=128000
#SBATCH --gres=gpu:1
#SBATCH --partition=longrun

module load cuda/11.4
module load cudnn/8.2
module load openmpi
module list
nvidia-smi
nvcc --version

srun python training.py


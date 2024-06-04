#!/bin/bash
#SBATCH --job-name=nlp-baseline 
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 #--gpus=1
#SBATCH --output=baseline.out
#SBATCH --reservation=fri
#SBATCH --time=04:00:00

FILE=baseline.py

module load CUDA/12.1.1

srun singularity exec --nv ./containers/container-torch.sif python "baseline_run.py"
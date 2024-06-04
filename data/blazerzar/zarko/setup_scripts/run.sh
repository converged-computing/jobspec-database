#!/bin/bash
#SBATCH --job-name=pytorch
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2gb
#SBATCH --time=30:00
#SBATCH --partition=gpu
#SBATCH --gpus=1

srun singularity exec --nv ./containers/pytorch.sif python code/arso_to_dataframe.py
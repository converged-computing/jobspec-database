#!/bin/bash

#SBATCH --job-name=train-2048-mamba
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:A40:1
#SBATCH --error=%x.err
#SBATCH --output=%x.out

module load singularity

singularity exec --nv nvidia.sif bash -c "$(cat $JOB_NAME.sh)"

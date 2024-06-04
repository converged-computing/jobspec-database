#!/bin/bash
#SBATCH --job-name=TWA_HP
#SBATCH --output=log_TWA_HP%a
#SBATCH --error=log_TWA_HP%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --array=1-8

srun julia -t 48 TWA_HP.jl ${SLURM_ARRAY_TASK_ID}

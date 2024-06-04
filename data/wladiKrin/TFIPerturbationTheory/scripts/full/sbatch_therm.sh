#!/bin/bash
#SBATCH --job-name=EDTherm
#SBATCH --output=log_EDThermNew%a
#SBATCH --error=log_EDThermNew%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --array=1-12

srun julia Hfull_Bound_therm.jl ${SLURM_ARRAY_TASK_ID}

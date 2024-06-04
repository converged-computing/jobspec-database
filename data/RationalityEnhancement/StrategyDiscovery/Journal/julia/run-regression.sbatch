#!/usr/bin/env bash
#SBATCH --job-name=run-regression
#SBATCH --output=regression/%j.out
#SBATCH --time=1400
#SBATCH --cpus-per-task=40

module load julia
julia -p 40 regression.jl #$SLURM_ARRAY_TASK_ID

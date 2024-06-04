#!/bin/bash

# Number of nodes to allocate, always 1
#SBATCH --nodes=1
# Number of MPI instances (ranks) to be executed per node, always 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20gb
# Give job a reasonable name
#SBATCH --job-name=short_time_phase_diagram
# File name for standard output (%j will be replaced by job id)
#SBATCH --output=short_time_phase_diagram-%a_%A.out
# File name for error output
#SBATCH --error=short_time_phase_diagram-%a_%A.err

srun julia run/phase_diagram.jl $SLURM_ARRAY_TASK_ID $1
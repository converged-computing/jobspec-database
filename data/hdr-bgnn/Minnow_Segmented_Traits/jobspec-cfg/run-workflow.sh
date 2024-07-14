#!/bin/bash
#SBATCH --job-name=MinnowTraits
#SBATCH --time=03:00:00
# Runs the Minnow Traits snakemake workflow 

# Usage:
# 
# sbatch --account <SLURM-account-name> run-workflow.sh

# Setup Snakemake/sbatch to use same account as this job
export SBATCH_ACCOUNT=$SLURM_JOB_ACCOUNT

# Configure Snakemake to run up to 20 jobs at once
NUM_JOBS=20

snakemake \
    --jobs $NUM_JOBS \
    --profile slurm/ \
    --use-singularity \
    --use-conda \
    "$@"

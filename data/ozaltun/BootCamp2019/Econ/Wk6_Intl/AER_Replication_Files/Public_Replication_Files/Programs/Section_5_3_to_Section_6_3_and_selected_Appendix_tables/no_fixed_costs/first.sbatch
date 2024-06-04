#!/bin/bash

#SBATCH --array=1-10
#SBATCH --job-name=first
#SBATCH --output=first_%A_%a.out
#SBATCH --error=first_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=sandyb
#SBATCH --time=36:00:00
#SBATCH --constraint=ib

module load matlab/2013b

# Create a local work directory
mkdir -p /tmp/tintelnot/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID

# Kick off matlab
matlab -nodisplay < Main_search1.m

# Cleanup local work directory
rm -rf /tmp/tintelnot/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID



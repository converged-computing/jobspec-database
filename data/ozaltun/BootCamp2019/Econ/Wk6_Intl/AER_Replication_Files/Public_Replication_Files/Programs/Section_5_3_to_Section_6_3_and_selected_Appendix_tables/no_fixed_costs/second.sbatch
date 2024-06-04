#!/bin/bash

#SBATCH --job-name=second
#SBATCH --output=second_%A.out
#SBATCH --error=second_%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=36:00:00
#SBATCH --partition=sandyb
#SBATCH --constraint=ib

module load matlab/2013b

# Create a local work directory
mkdir -p /tmp/tintelnot/$SLURM_JOB_ID

# Kick off matlab
matlab -nodisplay < Main_stat_counter2.m

# Cleanup local work directory
rm -rf /tmp/tintelnot/$SLURM_JOB_ID
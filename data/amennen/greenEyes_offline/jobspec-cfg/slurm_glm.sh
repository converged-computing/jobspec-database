#!/bin/bash

# Run within BIDS code/ directory:
# sbatch slurm_glm.sh

# Set current working directory
#SBATCH --workdir=.

# Set partition
#SBATCH --partition=all

# How long is job (in minutes)?
#SBATCH --time=60

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=1 --mem-per-cpu=4000

# Name of jobs?
#SBATCH --job-name=3dTproject

# Where to output log files?
#SBATCH --output='../derivatives/logs/glm_both_%A_%a.log'

# Number jobs to run in parallel
#SBATCH --array=31,39,53,68,81,88-122

# Remove modules because Singularity shouldn't need them
echo "Purging modules"
module purge
module load afni

# Print job submission info
echo "Slurm job ID: " $SLURM_JOB_ID
echo "Slurm array task ID: " $SLURM_ARRAY_TASK_ID
date

# Set subject ID based on array index
printf -v subj "%03d" $SLURM_ARRAY_TASK_ID

# Run fMRIPrep script with participant argument
echo "Running fMRIPrep on sub-$subj"

./run_glm.py $subj

echo "Finished running fMRIPrep on sub-$subj"
date

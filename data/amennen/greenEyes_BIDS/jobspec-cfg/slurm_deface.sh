#!/bin/bash

# Run within BIDS code/ directory: sbatch slurm_deface.sh

# Name of job?
#SBATCH --job-name=deface

# Set partition
#SBATCH --partition=all

# How long is job?
#SBATCH -t 1:00:00

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=8 --mem-per-cpu=20000
#SBATCH --array=2-14,16-19,25-26,28-33,35-46
# Where to output log files?
#SBATCH --output='../derivatives/deface/logs/deface-%A_%a.log'
#SBATCH --mail-user=amennen@princeton.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Remove modules because Singularity shouldn't need them
echo "Purging modules"
module purge

# Print job submission info
echo "Slurm job ID: " $SLURM_JOB_ID
echo "Slurm array task ID: " $SLURM_ARRAY_TASK_ID
date

# Set subject ID based on array index
printf -v subj "%03d" $SLURM_ARRAY_TASK_ID

# Run fMRIPrep script with participant argument
echo "Running pydeface on sub-$subj"

./deface.sh $subj

echo "Finished running pydeface on sub-$subj"
date

#!/usr/bin/env bash

#SBATCH -J 'array-example'
#SBATCH -o  log-array-example-%j.out
#SBATCH -p Brody
#SBATCH --time 00:10:00
#SBATCH --mem 1000
#SBATCH -c 1

# load the julia module
module load julia/1.2.0

# print out some info for the log file
echo "Slurm Job ID, unique: $SLURM_JOB_ID"
echo "Slurm Array Task ID, relative: $SLURM_ARRAY_TASK_ID"

# call the julia script which will start the farm
julia array_example.jl $1 $SLURM_JOB_ID $SLURM_ARRAY_TASK_ID

# move the log file to an input-dependent location
mv log-array-example-${SLURM_JOB_ID}.out $1/log-array-example-${SLURM_JOB_ID}.out

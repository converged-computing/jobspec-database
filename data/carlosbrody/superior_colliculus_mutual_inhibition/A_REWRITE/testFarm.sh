#!/usr/bin/env bash

#SBATCH -J 'testFarm'
#SBATCH -o  testFarm-log-%j.out
#SBATCH -p Brody
#SBATCH --time 96:00:00
#SBATCH -c 1

# load the julia module
module load julia/1.2.0

# print out some info for the log file
echo "Slurm Job ID, unique: $SLURM_JOB_ID"
echo "Slurm Array Task ID, relative: $SLURM_ARRAY_TASK_ID"

# call the julia script which will start the farm
arg1=$1
arg2=$2
shift
shift
sumofargs=$((arg1 + arg2))

echo "sum of args is $sumofargs"
echo "other args are $@"

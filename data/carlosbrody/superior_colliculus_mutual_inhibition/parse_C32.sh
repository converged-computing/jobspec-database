#!/usr/bin/env bash

#SBATCH -J 'parseC32'
#SBATCH -o log-parseC32-%j.out
#SBATCH -p Brody
#SBATCH --time 96:00:00
#SBATCH -c 1


# Start this script with
# sbatch --array=0-9 ./start_farm_spockC32.sh 

# load the julia module
module load julia/0.6.3

# print out some info for the log file
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Slurm Array Task ID: $SLURM_ARRAY_TASK_ID"

# Call the julia script which will start the farm
julia C32_parse_finished.jl 


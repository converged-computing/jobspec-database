#!/bin/bash

# This is a job array to run a bootstrap run for 101 random draws from the posterior 1st stage distribution.
# Job 0 is the baseline.
# Submit script: array=0-100 to process all 101 1st stage bootstrap draws
# This script will fill the appropriate folders specified in the file
# Note that each bootstrap run does 100 boostrap estimations and so may still take hours

#SBATCH --array=0-100             # there is a max array size - 512 tasks 
#SBATCH -J bootstrap
#SBATCH -p normal
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 2-00:00:00
#SBATCH --mem=32gb
#SBATCH -o logs/julia-%A-%a.out

# Load julia
module load julia/1.7.3
module load knitro/12.1.1

echo $(which julia);
echo $(which knitro);

# Make a log folder if it doesn't exist
mkdir -p logs

# Run julia with a command line arg
srun julia --project CARA_bootstrap_cluster.jl $SLURM_ARRAY_TASK_ID > logs/$SLURM_ARRAY_TASK_ID.txt

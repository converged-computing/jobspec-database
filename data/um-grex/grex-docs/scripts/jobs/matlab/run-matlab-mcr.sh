#!/bin/bash

#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=0-3:00:00
#SBATCH --job-name=Matlab-mcr-job
#SBATCH --partition=compute

# Choose the MCR directory according to the compiler version used.
# The one below for uofm/matlab/R2017A

MCR=/global/software/matlab/mcr/v93

# If running on Grex, uncomment the following line to set MCR_CACHE_ROOT:

module load mcr/mcr

echo "Running on host: `hostname`"
echo "Current working directory is `pwd`"
echo "Starting run at: `date`"

./run_mycode.sh $MCR > mycode_${SLURM_JOBID}.out

echo "Program finished with exit code $? at: `date`"

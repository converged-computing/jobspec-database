#!/bin/bash
# Set the name of the job
#SBATCH --job-name plinkinetti
# Launch an array of 100 jobs
## SBATCH --array 1-10
# Specify a time limit
## SBATCH --time 10:00:00
# Redirect stderr and stdout to the same file:
# %A will be replaced by the job ID and %a by the array index
#SBATCH -o plinkinetti_%A.out
#SBATCH -e plinkinetti_%A.out
# Send email notifications
## SBATCH --mail-type=ALL
# We request an exclusive node for every job in the array
## SBATCH --exclusive
# and reserve 10GB of memory
#SBATCH --mem=10000
# Specify the number of tasks (processes)
#SBATCH --ntasks 1
# Our job is multithreaded, so we ask 4 CPUs per process
#SBATCH --cpus-per-task=12
# So, in total we will have 4x8 CPUs for us
## module load nest/2.10.0
##################################################################
# from here on we can run whatever command we want (e.g., srun python simulate.py $SLURM_ARRAY_JOB_ID)
# we use slurm's environment variables to create unique output files and echo the name of the executing node in that file
##module load anaconda
##source activate r
srun Rscript job.R

#!/bin/bash
#SBATCH --account=def-pelleti2
#SBATCH --time=6-23:55           # time (DD-HH:MM)
#SBATCH --job-name=v4Cost # the name of the model
#SBATCH --array=1-3  # to run 3 chains 
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=35G                 # Default memory per CPU is 3GB.
#SBATCH --mail-user=gabriel.pigeon@usherbrooke.ca
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# If you would like to use more please adjust this.

## Below you can put your scripts
# If you want to load module

module load r/4.1.2

# module list # List loaded modules

# Other commands can be included below
cd ~/projects/def-pelleti2/pigeonga/Trsw_justine/

Rscript --verbose R/4_runModel.R $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID


# R CMD BATCH R/313_RMCMC.R
# salloc --time=1:0:0 --ntasks=1 --account=def-pelleti2

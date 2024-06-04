#!/bin/bash

#
# In order to submit this job to Slurm, run the command below from the terminal.
#   sbatch submit.sh

#SBATCH --time=01:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --constraint=48core
#SBATCH --cpus-per-task=48
#SBATCH --mem=24G
#SBATCH --job-name prob293 
#SBATCH --output prob293-%j.out
#SBATCH --error prob293-%j.err

#-------- End of Slurm commands -------------------------

module load rust 

cargo run --release

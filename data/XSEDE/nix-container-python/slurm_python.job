#!/bin/bash
#SBATCH -N 1 #Number of nodes
#SBATCH -n 4 #Number of "tasks"
#SBATCH -p cloud #Run in the "cloud" partition
#SBATCH -o sing_test_%A.out #The %A refers to the slurm job ID, this is useful for distinguishing output files

module purge
module load singularity

singularity run nix-container-python.sif

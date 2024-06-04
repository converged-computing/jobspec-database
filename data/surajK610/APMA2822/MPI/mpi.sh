#!/bin/bash

# Request an hour of runtime:
#SBATCH --time=1:00:00

# Use 2 nodes with 8 tasks each, for 16 MPI tasks:
#SBATCH -N 2
#SBATCH --tasks-per-node=8
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=1

# Specify a job name:
#SBATCH -J MyMPIJob

# Specify an output file
#SBATCH -o MyMPIJob-%j.out
#SBATCH -e MyMPIJob-%j.out

# Run a command
#srun --mpi=list
#srun --cpu-bind=help

srun --mpi=pmix_v4 ex1


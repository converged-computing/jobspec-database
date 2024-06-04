#!/bin/bash

#SBATCH -J mpi-diffusion
#SBATCH -t 0-0:60:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --mem-per-cpu=16GB
#SBATCH --mail-user=jpsamaroo@gmail.com
#SBATCH --output=output.log

export UCX_ERROR_SIGNALS="SIGILL,SIGBUS,SIGFPE"

module list
srun --mpi=pmi2 julia --project main.jl

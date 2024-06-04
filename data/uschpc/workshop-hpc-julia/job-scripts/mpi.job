#!/bin/bash

#SBATCH --account=<project_id>
#SBATCH --partition=epyc-64
#SBATCH --constraint=epyc-7513
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=1:00:00

module purge
module load julia/1.10.2
module load gcc/12.3.0
module load openmpi/4.1.6

ulimit -s unlimited

srun --mpi=pmix_v2 -n $SLURM_NTASKS julia mpi.jl

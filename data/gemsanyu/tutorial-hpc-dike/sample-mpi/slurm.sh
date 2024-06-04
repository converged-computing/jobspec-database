#!/bin/bash
#
#SBATCH --job-name=test_mpi
#SBATCH --output=result_mpi.txt
#
#SBATCH --nodes=1
#SBATCH --time=01:00
#SBATCH --ntasks-per-node=20 ### Number of tasks (MPI processes)/ number of processor max
srun --mpi=pmix hello.mpi

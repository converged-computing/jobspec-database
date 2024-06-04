#!/bin/sh
#SBATCH -t 10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p normal
#SBATCH -A proj_1371
#SBATCH --time=00:30:00
#SBATCH --constraint=type_a

module load openmpi/4.1.4
module load CUDA/11.7
module load lammps/2022sep15_cpu

srun --mpi=pmix_v2 lmp -i in.eam

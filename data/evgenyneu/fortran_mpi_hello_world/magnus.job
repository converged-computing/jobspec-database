#!/bin/bash -l
#SBATCH --job-name=hello_mpi_01
#SBATCH --account=ew6
#SBATCH --partition=workq
#SBATCH --time=00:00:10
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --export=NONE
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sausageskin@gmail.com
module swap PrgEnv-gnu PrgEnv-intel
module swap PrgEnv-cray PrgEnv-intel
srun --export=all ./build/hello_mpi

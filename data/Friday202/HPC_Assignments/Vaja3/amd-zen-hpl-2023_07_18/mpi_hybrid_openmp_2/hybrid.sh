#!/bin/bash
#SBATCH --reservation=fri
#SBATCH --job-name=hybrid
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --output=mpi_hybrid_openmp_2/hybrid.log

module load OpenMPI/4.1.5-GCC-12.3.0

export UCX_TLS=self, tcp, HPL_RAM_CAP=1.0

unset OMPI_MCA_osc

NT=32
NR=2
MAP_BY=socket

mpirun --map-by ${MAP_BY}:PE=$NT -np $NR -x OMP_NUM_THREADS=$NT -x OMP_PROC_BIND=spread -x OMP_PLACES=cores ./xhpl -p -t -f mpi_hybrid_openmp_2/hybrid.dat
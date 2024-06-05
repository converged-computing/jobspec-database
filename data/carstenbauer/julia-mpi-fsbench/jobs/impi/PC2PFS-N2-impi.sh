#!/bin/bash -l

#SBATCH -t 00:10:00
#SBATCH -N 2
#SBATCH --exclusive
#SBATCH -n 128
#SBATCH -c 2
#SBATCH -A pc2-mitarbeiter
#SBATCH -p normal
#SBATCH -q cont
#SBATCH -o PC2PFS-N2-impi.out

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
# export OMPI_MCA_hwloc_base_report_bindings=false

export JULIA_DEPOT_PATH=/scratch/pc2-mitarbeiter/bauerc/.julia_fsbench
export JULIA_MPI_BINARY=system

ml lang
ml Julia
ml load mpi/impi/2021.5.0-intel-compilers-2022.0.1 

echo "starting N 2 trials"
echo "Julia depot located at $JULIA_DEPOT_PATH"
for i in {1..5}
do
   time srun --cpu_bind=cores julia --project=/scratch/pc2-mitarbeiter/bauerc/devel/julia-mpi-fsbench /scratch/pc2-mitarbeiter/bauerc/devel/julia-mpi-fsbench/bench.jl
   # time srun --cpu_bind=cores julia --project=/scratch/pc2-mitarbeiter/bauerc/devel/julia-mpi-fsbench /scratch/pc2-mitarbeiter/bauerc/devel/julia-mpi-fsbench/bench.jl verbose
   echo "N 2 trial $i completed"
   sleep 10
done



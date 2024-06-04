#!/usr/bin/env bash

#SBATCH --job-name=intelmpi-singularity
#SBATCH --account=use300
#SBATCH --partition=compute
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=24
#SBATCH --time=00:30:00
#SBATCH --output=intelmpi-singularity.o%j.%N

declare -xr COMPILER_MODULE='intel/2018.1.163'
declare -xr MPI_MODULE='intelmpi/2018.1.163'

module purge
module load "${COMPILER_MODULE}"
module load "${MPI_MODULE}"
module list
export SINGULARITYENV_PREPEND_PATH=/opt/intel/2018.1.163/compilers_and_libraries_2018.1.163/linux/mpi/intel64/bin:/opt/intel/2018.1.163/compilers_and_libraries_2018.1.163/linux/bin/intel64
export SINGULARITYENV_LD_LIBRARY_PATH=/etc/libibverbs.d:/opt/intel/2018.1.163/compilers_and_libraries_2018.1.163/linux/mpi/intel64/lib:/opt/intel/2018.1.163/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/opt/intel/2018.1.163/compilers_and_libraries_2018.1.163/linux/ipp/../compiler/lib/intel64:/opt/intel/2018.1.163/compilers_and_libraries_2018.1.163/linux/ipp/lib/intel64:/opt/intel/2018.1.163/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64:/opt/intel/2018.1.163/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.4
printenv

time -p mpirun singularity exec --bind /etc,/opt,/oasis,/scratch centos.simg ./hello_world_intel_intelmpi.x

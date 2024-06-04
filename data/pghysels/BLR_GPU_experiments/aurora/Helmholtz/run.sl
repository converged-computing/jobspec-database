#!/bin/bash
#PBS -A CSC250STMS08_CNDA
#PBS -N suitesparse
#PBS -l select=8
#PBS -l walltime=0:30:00
#PBS -k doe
#PBS -l place=scatter
#PBS -q EarlyAppAccess

module use /soft/modulefiles
module load spack-pe-oneapi
module load oneapi/eng-compiler
module load oneapi/release #/2023.12.15.001
module load parmetis

NNODES=`wc -l < $PBS_NODEFILE`

NRANKS=6 # Number of MPI ranks to spawn per node

NDEPTH=8 # Number of hardware threads per rank (i.e. spacing between MPI ranks)
NTHREADS=8 # Number of software threads per rank to launch (i.e. OMP_NUM_THREADS)
NTOTRANKS=$(( NNODES * NRANKS ))

export ZES_ENABLE_SYSMAN=1

mpirun0="mpiexec -n ${NTOTRANKS} --ppn ${NRANKS} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} --env  OMP_PLACES=threads /soft/tools/mpi_wrapper_utils/gpu_tile_compact.sh"

for k in 100 150 200 250 300 350 400; do
    $mpirun0 /home/ghysels/BLR_GPU_experiments/aurora/code/STRUMPACK/build/examples/sparse/testHelmholtz $k \
	     --sp_enable_gpu \
	     > ~/HH/mpi${NTOTRANKS}_${k}_SYCL_GPU.log
done

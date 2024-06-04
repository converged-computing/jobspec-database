#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:09:59
#PBS -q debug
#PBS -l filesystems=home
#PBS -A CSC250STPM09
#PBS -k doe
#PBS -N nccl-test
#PBS -o nccl-test.out
#PBS -e nccl-test.error


module load gcc
module load cudatoolkit-standalone/11.8.0
export MPI_HOME=/opt/cray/pe/mpich/8.1.16/ofi/gnu/9.1
export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-11.8.0/

export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${MPI_HOME}/lib:$LD_LIBRARY_PATH

# Set location to store NCCL_TEST source/repository


export NCCL_TEST_HOME="/home/yuke/lyd/nccl-tests"

# Set location to store NCCL-PROFILE source/repository
NCCL_SRC_LOCATION="/home/yuke/lyd/nccl"
export NCCL_SRC_LOCATION

export LD_LIBRARY_PATH=${NCCL_SRC_LOCATION}/build/lib/:$LD_LIBRARY_PATH

mpiexec -n 4 --ppn 4 --cpu-bind core ${NCCL_TEST_HOME}/build/all_reduce_perf -b 512MB -e 512MB -f 2 -g 1
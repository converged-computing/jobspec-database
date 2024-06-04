#!/usr/bin/env bash
#PBS -q main
#PBS -A SCSG0001
#PBS -j oe
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=2:mpiprocs=2:ngpus=2

. config_env_openmpi.sh || exit 1


# Enable GPU support in the MPI library
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1

export MPI_ARGS="-n 2"

export CUDA_VISIBLE_DEVICES=0,1,2,3
make run_hello





# use MPI local rank internally to set the CUDA device, not anything from the environment
unset CUDA_VISIBLE_DEVICES

# default compiler
## run CPU case regardless of where we are
make --no-print-directory run_CPU
## run GPU cases only on GPU nodes
nvidia-smi -L >/dev/null && make --no-print-directory run_GPUd run_GPUm

echo "Done at $(date)"

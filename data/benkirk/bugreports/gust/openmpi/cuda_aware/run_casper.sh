#!/usr/bin/env bash
#PBS -q casper
#PBS -A SCSG0001
#PBS -j oe
#PBS -l walltime=00:30:00
#PBS -l select=2:ncpus=1:mpiprocs=1:ngpus=1
#PBS -l gpu_type=v100
. config_env_openmpi-casper.sh || exit 1

# use MPI local rank internally to set the CUDA device, not anything from the environment
unset CUDA_VISIBLE_DEVICES

export MPI_ARGS="-n 2"

# default compiler
## run CPU case regardless of where we are
make --no-print-directory run_CPU
## run GPU cases only on GPU nodes
nvidia-smi -L >/dev/null && make --no-print-directory run_GPUd run_GPUm

echo "Done at $(date)"

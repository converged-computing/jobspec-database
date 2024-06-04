#!/bin/bash
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH -t 0:05:00
#SBATCH -J tf2-benchmark-pm
#SBATCH -d singleton
#SBATCH -o logs/%x-%j.out

module list

# DEBUGGING SETTINGS
#export NCCL_DEBUG=INFO
#export TF_CPP_MIN_LOG_LEVEL=3
#export TF_CPP_MIN_VLOG_LEVEL=1
#export HOROVOD_LOG_LEVEL=trace
#export MPICH_RDMA_ENABLED_CUDA=0
#export MPICH_MAX_THREAD_SAFETY=multiple
#export MPICH_GPU_SUPPORT_ENABLED=1
#export MPIR_CVAR_GPU_EAGER_DEVICE_MEM=0

set -x
srun -l -u python horovod/examples/tensorflow2/tensorflow2_synthetic_benchmark.py

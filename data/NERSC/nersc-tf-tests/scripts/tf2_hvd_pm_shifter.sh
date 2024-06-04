#!/bin/bash
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH -t 0:05:00
#SBATCH -J tf2-benchmark-pm-shifter
#SBATCH -d singleton
#SBATCH -o logs/%x-%j.out
#SBATCH --module=gpu,nccl-2.15

set -x

srun -l -u --mpi=pmi2 shifter \
    bash -c "python horovod/examples/tensorflow2/tensorflow2_synthetic_benchmark.py"

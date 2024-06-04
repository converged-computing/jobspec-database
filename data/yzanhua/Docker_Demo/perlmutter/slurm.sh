#!/bin/bash

#SBATCH -A m844
#SBATCH -t 00:5:00
#SBATCH -N 2
#SBATCH -C cpu
#SBATCH --qos=debug

#SBATCH -o qout.256.%j
#SBATCH -e qout.256.%j


if test "x$SLURM_NTASKS_PER_NODE" = x ; then
   SLURM_NTASKS_PER_NODE=128
fi


NUM_NODES=$SLURM_JOB_NUM_NODES

NP=$((NUM_NODES * SLURM_NTASKS_PER_NODE))

ulimit -c unlimited

# -w /work: set the working directory inside the container to /work
# -v $PWD:/work: mount the current directory (where you submit the job) to /work (inside the container)
# --rm: remove the container after the job is done
# --mpi: use NERSC's MPI implementation
# yzanhua/perlmutter_img:latest: the name of the container image
# python3 hello.py: the command to run inside the container
srun -n $NP podman-hpc run -w /work -v $PWD:/work --rm --mpi yzanhua/perlmutter_img:latest python3 hello.py

# some other flags:
# --mpi	Uses optimized Cray MPI
# --cuda-mpi	Uses CUDA-aware optimized Cray MPI
# --gpu	Enable NVIDIA GPU
# --cvmfs	Enable the CVMFS filesystem
# Note that the --cuda-mpi flag must be used together with the --gpu flag.

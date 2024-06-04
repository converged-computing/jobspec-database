#!/bin/bash -l
#SBATCH --job-name="wave3D"
#SBATCH --output=log/wave3D.%j.o
#SBATCH --error=log/wave3D.%j.e
#SBATCH --time=1:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account class04

module load daint-gpu
module load Julia/1.9.3-CrayGNU-21.09-cuda
export MPICH_RDMA_ENABLED_CUDA=0
export IGG_CUDAAWARE_MPI=0
nvidia-smi
/scratch/snx3000/julia/class222/daint-gpu/bin/mpiexecjl  -n 4 julia --project -O2 src/wave3D_multixpu.jl

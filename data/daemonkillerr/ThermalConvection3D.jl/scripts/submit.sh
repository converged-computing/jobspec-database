#!/bin/bash -l
#SBATCH --job-name="3D_Lava_Lamp"
#SBATCH --output=my_gpu_run.%j.o
#SBATCH --error=my_gpu_run.%j.e
#SBATCH --time=08:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account class04

module load daint-gpu
module load Julia/1.9.3-CrayGNU-21.09-cuda

export MPICH_RDMA_ENABLED_CUDA=0
export IGG_CUDAAWARE_MPI=0

srun -n8 bash -c 'julia -O3 ThermalConvection3D.jl'

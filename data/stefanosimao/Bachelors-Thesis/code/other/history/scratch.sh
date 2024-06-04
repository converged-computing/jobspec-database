#!/bin/bash -l

#SBATCH --job-name=cnn
#SBATCH --time=00:30:00
#SBATCH --nodes=10
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
#SBATCH --constraint=gpu
#SBATCH --account=c24
#SBATCH --output=m10.out
#SBATCH --error=m10.err

module load daint-gpu
module load PyTorch

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=ipogif0
export NCCL_IB_CUDA_SUPPORT=1

srun python main.py -bs ${1}
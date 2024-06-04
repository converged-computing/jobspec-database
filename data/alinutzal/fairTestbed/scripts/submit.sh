#!/bin/sh

#SBATCH --gpus-per-node=1
#SBATCH --account=pls0144
#SBATCH --time=05:00:00
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task 20
##SBATCH --constraint=48core 
#SBATCH -J mnist
#SBATCH -o alazar-%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=alazar@ysu.edu

module load miniconda3
module load cuda/11.8.0
source activate torch
srun --gpu_cmode=exclusive NCCL_P2P_LEVEL=NVL python 3_mnist_pl.py
#!/bin/bash
#SBATCH -p akya-cuda
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=00-08:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahmetoglu.alper@gmail.com

# module load /truba/home/aahmetoglu/privatemodules/cuda_10.0_module

echo "SLURM_NODELIST $SLURM_NODELIST"
echo "NUMBER OF CORES $SLURM_NTASKS"

nvidia-smi

python -u run_train.py


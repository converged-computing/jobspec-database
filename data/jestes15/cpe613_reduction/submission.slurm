#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 16
#SBATCH --mem=20G
#SBATCH --ntasks 1

# list NVIDIA cards, make clean, load the cuda module, make, and run main
lspci -vvv |& grep "NVIDIA" |& tee slurm-lspci.out

make A100 && \
  ./main_a100
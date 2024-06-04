#!/usr/bin/env zsh

#SBATCH --job-name=task2
#SBATCH --partition=instruction
#SBATCH --time=00-00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=task2.out


cd $SLURM_SUBMIT_DIR

module load nvidia/cuda/11.8

nvcc task2.cu reduce.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task2

./task2 10 1024

#./task2 {n} {threads_per_block}


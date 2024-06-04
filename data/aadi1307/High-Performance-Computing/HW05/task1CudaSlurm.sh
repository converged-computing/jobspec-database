#!/usr/bin/env zsh
#SBATCH --job-name=task1
#SBATCH --partition=instruction
#SBATCH --time=00-00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=task1.out
#SBATCH --mem=20G

module load nvidia/cuda/11.8.0

nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1

# Array of values
values=(32 64 128 256 512 1024 2048)

# Loop through each value and run the task
for val in "${values[@]}"; do
    ./task1 $val 16
done
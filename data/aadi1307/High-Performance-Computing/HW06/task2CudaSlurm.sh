#!/usr/bin/env zsh
#SBATCH --job-name=task2
#SBATCH --partition=instruction
#SBATCH --time=00-00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=task2.out
#SBATCH --mem=20G

module load nvidia/cuda/11.8.0

nvcc task2.cu scan.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task2

# Array of values
values=(1024 2048 4096 8192 16384 32768 65536)

# Loop through each value and run the task
for val in "${values[@]}"; do
    ./task2 $val 1024
done



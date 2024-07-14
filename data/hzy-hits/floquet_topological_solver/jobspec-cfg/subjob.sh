#!/bin/bash

# Request a GPU partition node and access to 1 GPU


# Ensures all allocated cores are on the same node
#SBATCH -N 1

# Request 1 CPU core
#SBATCH -n 32

#SBATCH -t 10:30:00
#SBATCH -o with_gpu.out
#SBATCH -e with_gpu.err

# Load CUDA module
module load gcc/10.2 cmake/3.15.4  ninja/1.9.0 eigen/3.4.0


cd ./
rm -rf data
mkdir -p data
rm -rf build
mkdir -p build
cd build

cmake .. -G Ninja
ninja


./Mat_Solver

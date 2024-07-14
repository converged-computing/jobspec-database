#!/bin/bash
#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:1 --gres-flags=enforce-binding

# Ensures all allocated cores are on the same node
#SBATCH -N 1

# Request 1 CPU core
#SBATCH -n 1

#SBATCH -t 01:30:00
#SBATCH -o with_gpu.out
#SBATCH -e with_gpu.err

# Load CUDA module
module load cuda/12.2.2  gcc/10.2   


nvidia-smi 

# Compile CUDA program and run
#nvcc -arch sm_20 vecadd.cu -o vecadd
nvcc -O2 -std=c++11 hw4.cu -o a.out
nvprof ./a.out
nsys profile ./a.out

#!/bin/bash
#SBATCH --output=./output/JOB.out

# Runtime and memory
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#### Your shell commands below this line ####
JULIA_DEBUG=CUDA julia src/JOB.jl

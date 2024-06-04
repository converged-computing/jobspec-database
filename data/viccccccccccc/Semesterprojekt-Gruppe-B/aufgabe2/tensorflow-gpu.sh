#!/bin/bash

# Job Name
#SBATCH --job-name=tensorflow-gpu
# Number of Nodes
#SBATCH --nodes=1
# Set the GPU-Partition (opt. but recommended)
#SBATCH --partition=gpu
# Allocate node with certain GPU
#SBATCH --gres=gpu:gtx745

module load cuda

python versuch2.1.py
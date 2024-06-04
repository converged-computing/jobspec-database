#!/bin/bash

# Job name
#SBATCH --job-name=fold2array

#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus=1

# Partition
#SBATCH --partition=gpu
#SBATCH --mail-user=jake.kovalic@yale.edu   
#SBATCH --mail-type=ALL

# Expected running time
#SBATCH --time=12:00:00

# Output and error files
#SBATCH --output=adj_matrix_conversion_gpu.out
#SBATCH --error=adj_matrix_conversion_gpu.err

# Load necessary modules
module load miniconda
conda activate env_3_8

# Run the Python script
python fold2array_batch.py

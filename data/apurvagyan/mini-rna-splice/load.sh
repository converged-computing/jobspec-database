#!/bin/bash

#SBATCH --job-name=load_graphs_batch
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err


module load miniconda
conda activate env_3_8

python load_batch.py

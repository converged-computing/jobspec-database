#!/bin/bash

#SBATCH --job-name=autoencoder
#SBATCH --partition=msfea-ai

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu
#SBATCH --mem=0

module purge
module load python/tensorflow-2.3.1
module load cuda

python3 train_autoencoder.py

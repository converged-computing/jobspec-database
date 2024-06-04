#!/bin/bash

#SBATCH -c 10
#SBATCH -n 1   
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH --mem=100G
#SBATCH -G a100:1
#SBATCH -p general  
#SBATCH -q public
#SBATCH --mail-type=ALL

module load mamba/latest

source activate inr2array

python -m experiments.make_latent_dset --rundir ./outputs/2024-01-05/13-57-00 --output_path experiments/data/mnist-embeddings.pt

python -m experiments.launch_classify_latent embedding_path=experiments/data/mnist-embeddings.pt
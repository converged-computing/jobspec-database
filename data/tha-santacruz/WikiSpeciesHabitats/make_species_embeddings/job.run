#!/bin/bash -l

#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --partition=debug
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

# include other commands here that might be needed prior to launching

python make_species_doc2vec_species_embeddings.py
#!/bin/bash -l

#SBATCH --gres=gpu:titan-x:1
#SBATCH --mem=8G
#SBATCH -t 720
source activate tf
julia "$@"

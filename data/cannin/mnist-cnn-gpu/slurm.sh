#!/bin/bash

#SBATCH -c 4
#SBATCH -t 6:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:teslaK80:2
#SBATCH --mem-per-gpu=8G
#SBATCH --mail-user=augustin_luna@hms.harvard.edu
#SBATCH --mail-type=ALL
#SBATCH -o %j.out
#SBATCH -e %j.err

module load gcc/6.2.0
module load python/3.7.4
module load cuda/10.0

pipenv run python mnist_cnn_gpu.py

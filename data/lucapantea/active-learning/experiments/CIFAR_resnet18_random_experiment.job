#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=CIFAR10_resnet18_random_experiment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=../out/slurm_output_%x.out

module purge
module load 2022
module load Anaconda3/2022.05

# Create environment specific variables
source ../.env

# Activate conda environment
source activate $PROJECT_ENV_NAME

# Change to working directory
cd $PROJECT_DIR

# Ensure wandb online
wandb online

# Realistic values
n_init_labeled_realistic=100
num_valid_realistic=300
n_query_realistic=10

# # Ideal values
# n_init_labeled_ideal=
# num_valid_ideal=
# n_query_ideal=

python main.py --experiment \
               --debug \
               --strategy random \
               --dataset cifar10 \
               --model resnet18 \
               --epochs 100 \
               --n_round 100 \
               --num_valid $num_valid_realistic \
               --n_init_labeled $n_init_labeled_realistic \
               --n_query $n_query_realistic

# python main.py --experiment \
#                --debug \
#                --strategy random \
#                --dataset cifar10 \
#                --model resnet18 \
#                --epochs 100 \
#                --n_round 100 \
#                --num_valid $num_valid_ideal \
#                --n_init_labeled $n_init_labeled_ideal \
#                --n_query $n_query_ideal
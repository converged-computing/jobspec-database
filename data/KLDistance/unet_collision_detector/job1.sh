#!/bin/bash
#SBATCH --get-user-env=L
#SBATCH --job-name=beam_collision_discriminator_trainer1
#SBATCH --time=02:10:00
#SBATCH --mem=24G
#SBATCH --output=output.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4

python training.py
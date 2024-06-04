#!/bin/bash
#SBATCH --time=4:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8

#SBATCH --mem=60G
#SBATCH --partition=amdgpufast
#SBATCH --gres=gpu:1

#SBATCH --error=logs/train_flow_%j.out
#SBATCH --output=logs/train_flow_%j.out


ml torchsparse

cd $HOME

python -u motion_supervision/train.py


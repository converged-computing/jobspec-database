#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpuA100 
#SBATCH --time=02:00:00
#SBATCH --job-name=setup
#SBATCH --output=setup.out
 
# Set up environment
uenv verbose cuda-11.4 cudnn-11.4-8.2.4

python3 -m pip install -r requirements.txt --user

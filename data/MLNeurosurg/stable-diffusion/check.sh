#!/bin/bash

#SBATCH --job-name=data
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --mem-per-cpu=5g
# SBATCH --gres=gpu:0
#SBATCH --time=120:00:00
#SBATCH --account=precisionhealth_project1
#SBATCH --partition=standard
#SBATCH --mail-user=achowdur@umich.edu
#SBATCH --export=ALL

# python job to run
python check_data.py

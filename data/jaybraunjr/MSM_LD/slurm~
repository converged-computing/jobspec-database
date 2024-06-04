#!/bin/bash
#SBATCH --time=0-48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --job-name=msm
#SBATCH --out=msm.out
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mem=300G
#SBATCH --err=err.log


#SBATCH --account=swanson-gpu-np
#SBATCH --partition=swanson-gpu-np


python3 calling_func_v3.py

echo "... Job Finished at `date`"

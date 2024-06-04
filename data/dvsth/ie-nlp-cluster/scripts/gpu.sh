#!/bin/bash
#SBATCH -p compsci-gpu --gres=gpu:1 
#SBATCH -c 6 
#SBATCH --job-name=ieneurips
hostname
nvidia-smi --query-gpu=gpu_name,memory.total,memory.free --format=csv
source env/bin/activate
echo 'env started'
time python3 nlp.py
echo 'done'
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=20:05:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-gpu=4

cd /home/adnanzai/project/monai-train
ml purge
module load python/3.11 poetry cuda

source $(poetry env info --path)/bin/activate
python -m monai-ops --optuna ./example/optuna_config.yaml --data=/home/adnanzai/mice_data_v2 --output /home/adnanzai/optuna --seed 0


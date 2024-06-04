#!/bin/bash
#SBATCH --partition=prod
#SBATCH --gres=gpu:1
#SBATCH --job-name="ResNet18_FS_BW"
#SBATCH --array=1-1
#SBATCH --time=0:06:0

source venv/bin/activate

python main.py --config configs/resnet18_binary_classification.yaml --verbose
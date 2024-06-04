#!/bin/bash
#SBATCH -p class -A sxg125_csds438
#SBATCH --gres=gpu:1
#SBATCH --mem=100gb
#SBATCH -t 5:00:00

module load cuda/11.2
source venv/bin/activate

# Add extra arguments to the train script
python train.py \
    --model srresnet
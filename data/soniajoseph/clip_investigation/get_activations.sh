#!/bin/bash
#SBATCH --array=4-9
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=48Gb
#SBATCH --time=1:00:00
###SBATCH --ntasks=16
#SBATCH --output=get_activations.%A.%a.out
#SBATCH --error=get_activations.%A.%a.err
#SBATCH --job-name=get_activations

module load anaconda/3
module load cuda/11.7
module load libffi

source /home/mila/s/sonia.joseph/ViT-Planetarium/env/bin/activate

python get_activations.py --layer_num $SLURM_ARRAY_TASK_ID --attn
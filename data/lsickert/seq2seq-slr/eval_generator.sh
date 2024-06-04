#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64GB
#SBATCH --job-name=eval_generator_model
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.m.sickert@student.rug.nl

module purge

module load Python/3.8.6-GCCcore-10.2.0

source /data/$USER/.envs/seq2slr/bin/activate

module load PyTorch/1.10.0-fosscuda-2020b

# move the cached datasets to the /scratch directory so that we have more space available
export HF_DATASETS_CACHE="/scratch/$USER/.cache/huggingface/datasets"

# Change this to the correct file name
python -u main.py --action eval_generator --lang en --qual gold,silver --name t5-base

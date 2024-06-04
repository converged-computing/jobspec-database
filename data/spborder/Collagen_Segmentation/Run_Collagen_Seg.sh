#!/bin/sh
#SBATCH --qos=pinaki.sarder
#SBATCH --job-name=collagen_segmentation
#SBATCH --partition=gpu
#SBATCH --gpus=geforce:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80gb
#SBATCH --time=25:00:00
#SBATCH --output=collagen_seg_training_%j.out

pwd; hostname; date
module load singularity

ml
date
nvidia-smi

export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjNzllZGRmMC0yMzg2LTRhMzktOTk1MC1hNDc2MDlkNjVkYTMifQ=="


singularity exec --nv collagen_segmentation_latest.sif python3 Collagen_Segmentation/CollagenSegMain.py train_inputs_single.json

date
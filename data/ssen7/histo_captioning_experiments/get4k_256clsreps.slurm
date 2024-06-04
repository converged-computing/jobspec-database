#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=get_4kreps.out
#SBATCH -A gutintelligencelab
#SBATCH --partition=bii-gpu
#SBATCH --gres=gpu:v100:2

module load singularity pytorch/1.10.0
singularity run --nv $CONTAINERDIR/pytorch-1.10.0.sif /home/ss4yd/vision_transformer/captioning_vision_transformer/generate4k_256clsreps.py
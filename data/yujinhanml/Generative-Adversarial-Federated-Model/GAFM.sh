#!/bin/bash

#SBATCH --job-name=GAFM
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --output=output_GAFM.txt
#SBATCH --time=2-

#module load miniconda3
#module load cuDNN/7.6.2.24-CUDA-10.0.130
#module load CUDA
#module load cuDNN
#module load miniconda3
#module load cuDNN/7.6.2.24-CUDA-10.0.130 
#load python/3.9.12
python train.py











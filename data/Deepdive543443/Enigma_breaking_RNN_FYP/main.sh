#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --priority rse-com6012
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=6G
#SBATCH --mail-user=qfeng10@sheffield.ac.uk
#SBATCH --output=output.%j.test.out
#SBATCH --cpus-per-task=4

module load Anaconda3/5.3.0
module load cuDNN/7.6.4.38-gcccuda-2019b
source activate torch

python main.py --PROGRESS_BAR=0 --USE_COMPILE=1 --LOG='tensorboard/16HEAD' --ATTN_HEAD=16
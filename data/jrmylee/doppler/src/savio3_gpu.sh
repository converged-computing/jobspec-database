#!/bin/bash
# Job name:
#SBATCH --job-name=daniil
#
# Account:
#SBATCH --account=fc_deepmusic
#
# Partition:
#SBATCH --partition=savio3_gpu
#
# Number of nodes:
#SBATCH --nodes=1
#
#SBATCH -q v100_gpu3_normal
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=8
#
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=1
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:V100:2
#
# Wall clock limit:
#SBATCH --time=30:00:00
#
## Command(s) to run (example):
module unload python/3.7
module load ml/tensorflow/2.5.0-py37 libsndfile
python train.py

#!/bin/bash
# FILENAME: guo.sub

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -t 2:00:00
#SBATCH -A guo675-h

module purge
module load anaconda
module load anaconda/2020.11-py38
module load cuda/11.7.0
module load cudnn/cuda-11.7_8.6
module load use.own
module load conda-env/my_tf_env-py3.8.5

python control.py

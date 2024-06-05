#!/bin/bash

#SBATCH --job-name=wmt-en2de

#SBATCH --qos=qos_gpu-t4

#SBATCH --output=./logfiles/logfile_wmt_s2s.out

#SBATCH --error=./logfiles/logfile_wmt_s2s.err

#SBATCH --time=48:00:00

#SBATCH --ntasks=1

#SBATCH --gres=gpu:4

#SBATCH --cpus-per-task=40

#SBATCH --hint=nomultithread

#SBATCH --constraint=v100-32g


module purge
module load anaconda-py3/2019.03
conda activate modelcomparisontranslation
set -x
nvidia-smi
# This will create a config file on your server


srun accelerate launch --multi_gpu train_mp_transformer.py
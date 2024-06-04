#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=small
#SBATCH --job-name=HopfieldLM
#SBATCH --gres=gpu:1

module load cuda/9.2

echo $CUDA_VISIBLE_DEVICES
nvidia-smi
echo $PWD
# run the application
python3 main.py -m lstm -b 128  -d 1mb -g 0 > slurm-lstmLM-1mb-$SLURM_JOB_ID.out

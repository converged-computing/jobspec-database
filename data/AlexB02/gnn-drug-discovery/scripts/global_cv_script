#!/bin/bash
#PBS -l select=1:ncpus=8:mem=24gb:ngpus=1:gpu_type=RTX6000
#PBS -lwalltime=00:30:00

module load anaconda3/personal

echo $CUDA_VISIBLE_DEVICES

cd $HOME/gnn-drug-discovery/
source activate gnndd-cuda
WANDB__SERVICE_WAIT=300 CUDA_LAUNCH_BLOCKING=1 python main.py global_cv

#!/bin/bash
#SBATCH --partition=teaching
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=11
#SBATCH --output=pong_job.out
#SBATCH --job-name=pong_selfplay_batch21

# Your job
#python3 -m pip3 install --user -r requirements.txt
singularity run --nv /data/containers/msoe-tensorflow-20.07-tf2-py3.sif python3 reinforcement_selfplay.py

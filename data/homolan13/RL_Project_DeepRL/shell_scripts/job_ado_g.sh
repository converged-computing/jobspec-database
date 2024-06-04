#!/bin/bash

# You must specify a valid email address!
#SBATCH --mail-user=yanis.schaerer@students.unibe.ch

# Mail on NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-type=FAIL,END

# Job name 
#SBATCH --job-name="g_ado"

# Runtime and memory
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=2G

#SBATCH --cpus-per-task=8

# Partition
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx2080ti:2

# Install dependencies #
singularity exec --nv docker://pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime pip install -U -e simglucose_local # Gym will also be installed

# Run script #
singularity exec --nv docker://pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime python g_training_adolescent.py
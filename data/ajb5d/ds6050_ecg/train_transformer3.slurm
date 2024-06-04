#!/bin/bash
#SBATCH -A ds6050
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100
#SBATCH --constraint=a100_80gb
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00

module load singularity tensorflow/2.10.0
singularity run --nv $CONTAINERDIR/tensorflow-2.10.0.sif transformer_age3.py
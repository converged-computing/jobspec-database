#!/bin/bash
#SBATCH -J PrimeNet
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=10GB
#SBATCH -t 200:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output ./Out/PrimeNet.out


python -u ./main.py

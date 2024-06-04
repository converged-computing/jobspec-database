#!/bin/bash
#SBATCH --job-name=Q1
#SBATCH --account=macs30113
#SBATCH --ntasks=40
#SBATCH --nodes=10
#SBATCH --time=00:10:00
#SBATCH --partition=broadwl
#SBATCH --output=Q1.out
#SBATCH --mem-per-cpu=30G


module load python 

module load cuda

python finished.py

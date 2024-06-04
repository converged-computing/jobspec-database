#!/bin/bash
#SBATCH -J Super
#SBATCH -N 1
#SBATCH --cpus-per-task=10
#SBATCH -t 24:45:00
#SBATCH --mem=32GB
#SBATCH --partition=amd
module load python/3.8.6
cd $HOME/AutoML
python autocausality_AutoSuper.py

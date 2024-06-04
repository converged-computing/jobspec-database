#!/bin/bash
#PBS -P q27
#PBS -q gpuvolta
#PBS -l walltime=05:00:00,ncpus=12,ngpus=1,mem=300GB,jobfs=10GB
#PBS -l storage=scratch/q27
#PBS -l wd

module load python3/3.10.4
module load cuda/12.2.2

python3 runCase.py 8 100

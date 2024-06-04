#!/bin/bash
#PBS -P ldrd
#PBS -q gpu
#PBS -N PINN
#PBS -l select=1:ncpus=12:ngpus=4:mem=20GB
#PBS -l walltime=12:00:00
#PBS -M syang@caps.fsu.edu
#PBS -m ae

module load cuda
module load tensorflow
 
cd "$PBS_O_WORKDIR"
python HX.py


#!/bin/bash

#SBATCH -o mapsim_%j.out
#SBATCH -e mapsim_%J.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=scottgbarnes@gwu.edu

#SBATCH -N 1
#SBATCH -p debug-cpu

#SBATCH -D /home/scottgbarnes/Cooperative-Search-Gym
#SBATCH -J mapsim

#SBATCH -t 4:00:00

module load anaconda
source activate tensorflow
python run_mapEnv.py

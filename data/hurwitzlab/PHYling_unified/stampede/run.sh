#!/bin/bash

#SBATCH -J phyling
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p normal
#SBATCH -t 24:00:00
#SBATCH -A iPlant-Collabs

module load tacc-singularity

set -u

singularity run phyling.img

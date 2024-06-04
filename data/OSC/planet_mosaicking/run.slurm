#!/bin/bash
#SBATCH -J planetpy
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -A PZS0720
#SBATCH --exclusive

module load python/3.7-2019.10
export PATH=/users/PZS0530/skhuvis/opt/mambaforge/22.9.0-2/bin:$PATH #mamba
source activate s2s2

export PYTHONUNBUFFERED=TRUE

python ./mosaic.py

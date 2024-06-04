#!/bin/bash
#SBATCH -J test
#SBATCH -p workq2
##SBATCH -N 1
##SBATCH --ntask-per-node 1
#SBATCH -o ptest.out
#SBATCH -e ptest.err
module load herramientas/python/3.6

PATH=/home/mroldan/.conda/envs/carto/bin:$PATH

#conda activate carto
srun python plot_test.py
#conda deactivate

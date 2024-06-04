#!/bin/bash
#SBATCH -J planet          # job name
#SBATCH -o planet.o%j             # output file name (%j expands to jobID)
#SBATCH -e planet.e%j             # error file name (%j expands to jobID)
#SBATCH -n 224
#SBATCH -t 12:00:00             # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=adrn@princeton.edu
#SBATCH --mail-type=begin       # email me when the job starts
#SBATCH --mail-type=end         # email me when the job finishes

cd /tigress/adrianp/projects/dailyplanet/scripts/

module load openmpi/gcc/1.10.2/64

source activate twoface

date

srun python run_planetz.py -v -c ../config/planetz.yml --mpi --data-path=../data/keck_vels/ --ext=vels

date

#!/bin/bash
# Job name:
#SBATCH --job-name=test
#
# Account:
#SBATCH --account=fc_paciorek
#
# Partition:
#SBATCH --partition=savio2
#
# Number of tasks (e.g., MPI tasks, here 2 nodes worth)
#SBATCH --ntasks=48
#
# Wall clock limit (30 minutes here):
#SBATCH --time=00:30:00
#
## Command(s) to run:
module load python/3.6
python calc.py >& calc.out

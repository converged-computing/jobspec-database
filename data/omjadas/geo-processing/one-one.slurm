#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0-12:00:00
#SBATCH --partition=physical

module load Python/3.6.4-intel-2017.u2

time mpiexec -n 1 python3 app.py bigTwitter.json melbGrid.json
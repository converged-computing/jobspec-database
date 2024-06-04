#!/bin/bash
#SBATCH --ntasks=64 --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name dask_square
#SBATCH -o %x.%j.stdout
#SBATCH -e %x.%j.stderr

module --force purge
module load calcua/2020a
module load Python
module list

srun python dask_square.py

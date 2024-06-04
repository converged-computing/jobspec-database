#!/usr/bin/env bash
#SBATCH -p hebbe
#SBATCH -A C3SE2021-1-15
#SBATCH	-J GA
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 10:00:00
#SBATCH -o stdout
#SBATCH -e stderr

module purge
module load intel/2019a GPAW ASE

mpirun gpaw-python ga.py

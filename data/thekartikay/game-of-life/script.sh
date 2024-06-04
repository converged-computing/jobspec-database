#!/bin/bash

# set output and error output filenames, %j will be replaced by Slurm with the jobid
#SBATCH -o project_2_%j.out
#SBATCH -e project_2_%j.err 
#SBATCH -p c1exp
#SBATCH -N 1
#SBATCH -J PROJECT_2
#SBATCH -D /home/avi_kartikay/Project_2/AVI_KARTIKAY_DATS6402_10_PROJECT_2
# half hour timelimit
#SBATCH -t 01:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=avi_kartikay@gwu.edu
module load mpi4py
# module load singularity
# export LM_LICENSE_FILE="27000@localhost"

# test.m is your matlab code
mpirun -n 28 singularity exec --bind /groups --bind /lustre /groups/dats6402_10/images/python-3.7.0+ompi.simg python3 code.py


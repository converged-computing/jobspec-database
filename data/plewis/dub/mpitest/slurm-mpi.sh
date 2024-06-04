#!/bin/bash

#SBATCH --partition=priority
#SBATCH --constraint='skylake'
#SBATCH --qos=pol02003sky
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --job-name=snakempi
#SBATCH -o mpi-%j.out
#SBATCH -e mpi-%j.err

LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/lib"
export TIMEFORMAT="user-seconds %3U"
cd /home/pol02003/dub/mpitest
time mpirun -n 20 dubmpi

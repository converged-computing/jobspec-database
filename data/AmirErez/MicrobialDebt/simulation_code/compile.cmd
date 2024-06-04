#!/bin/bash
#SBATCH -N 1 # node count
#SBATCH -c 1
#SBATCH -t 18:00:0
#SBATCH --mem=4000
##SBATCH --ntasks-per-core=2
#SBATCH -o logs/compile_%A.out

module load matlab/2021a mcc
export MCR_CACHE_ROOT=/tmp/$SLURM_JOB_ID
mcc -mv automated_run_serialdil.m

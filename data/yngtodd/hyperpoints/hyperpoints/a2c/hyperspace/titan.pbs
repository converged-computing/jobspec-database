#!/bin/bash -l
#PBS -A CSC264
#PBS -l walltime=4:00:00
#PBS -l nodes=16
#PBS -N A2C 
#PBS -j oe

CSC264=/lustre/atlas/proj-shared/csc264
export SINGULARITYENV_MPICH_GNI_MALLOC_FALLBACK=1

cd $CSC264/yngtodd/hyperpoints

aprun -n 16 -N 1 $CSC264/singularity/basic_py36/bin/python optimize.py --results_dir results 

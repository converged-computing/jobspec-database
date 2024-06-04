#!/bin/bash
#PBS -A ARLAP35100034
#PBS -q standard
#PBS -l select=22:ncpus=48:mpiprocs=48
#PBS -l walltime=15:00:00
#PBS -N %T%%R%%PREF% 
#PBS -j oe
#PBS -V
#PBS -S /bin/bash
#PBS -m be
#PBS -M cfa22@drexel.edu

NAMD2=/app/namd/NAMD_2.14/Linux-x86_64-icc/namd2
CONF=%CONFIG%
LOG=%LOG%
cd $PBS_O_WORKDIR
mpiexec -n 1056 $NAMD2 $CONF > $LOG


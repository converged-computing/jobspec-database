#!/bin/bash
# This is a template qsub script for narwhal.navydsrc.hpc.mil
# Cameron F. Abrams cfa22@drexel.edu

#PBS -A ARLAP26313136
#PBS -q standard
#PBS -l select=6:ncpus=128:mpiprocs=1
#PBS -l walltime=15:00:00
#PBS -N %NAME%
#PBS -j oe
#PBS -V
#PBS -S /bin/bash
#PBS -m be
#PBS -M cfa22@drexel.edu

NAMD2=/app/namd/NAMD_2.14/bin_cpu/namd2
CONF=%CONF%
LOG=%LOG%
cd $PBS_O_WORKDIR
mpiexec -n 768 $NAMD2 $CONF > $LOG


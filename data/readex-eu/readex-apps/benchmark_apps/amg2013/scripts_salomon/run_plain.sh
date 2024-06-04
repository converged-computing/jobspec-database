#!/bin/bash

#PBS -A OPEN-12-63
#PBS -q qprod
#PBS -N amg2013_plain
#PBS -o amg2013_plain.out
#PBS -e amg2013_plain.err
#PBS -l select=4:ncpus=24:mpiprocs=2:ompthreads=12:accelerator=true
#PBS -l walltime=00:30:00
###############################################################################

if [ "$PBS_ENVIRONMENT" == "PBS_BATCH" ]; then
  LOC=$PBS_O_WORKDIR
else
  LOC=$(pwd)
fi

cd $LOC/..

module purge
source ./readex_env/set_env_plain.source

mpirun $BIND_TO_SOCKETS ./test/amg2013_plain -P 2 2 2 -r 40 40 40 


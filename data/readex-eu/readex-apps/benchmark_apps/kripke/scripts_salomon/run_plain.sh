#!/bin/sh

#PBS -A OPEN-12-63
#PBS -q qprod
##PBS -q qexp
#PBS -N compare_kripke
#PBS -l select=1:ncpus=24:mpiprocs=24:accelerator=false
#PBS -l x86_adapt=true
#PBS -l walltime=02:00:00
#PBS -m be

if [ "$PBS_ENVIRONMENT" == "PBS_BATCH" ]; then
	export FM_DIR=$PBS_O_WORKDIR
else
	export FM_DIR=$(pwd)
fi
cd $FM_DIR/../build

. ../readex_env/set_env_plain.source
. ../environment.sh

mpirun -n 24 ./kripke $KRIPKE_COMMAND


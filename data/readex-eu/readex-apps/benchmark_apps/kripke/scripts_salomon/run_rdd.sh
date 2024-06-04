#!/bin/sh

#PBS -A OPEN-12-63
#PBS -q qprod
#PBS -N KRIPKE_RDD
#PBS -l select=1:ncpus=24:mpiprocs=24:accelerator=false
#PBS -l x86_adapt=true
#PBS -l walltime=01:00:00
#PBS -m be

if [ "$PBS_ENVIRONMENT" == "PBS_BATCH" ]; then
	export FM_DIR=$PBS_O_WORKDIR
else
	export FM_DIR=$(pwd)
fi
cd $FM_DIR/../build

. ../readex_env/set_env_rdd.source
. ../environment.sh

export SCOREP_PROFILING_FORMAT=cube_tuple
export SCOREP_METRIC_PAPI=PAPI_TOT_INS,PAPI_L3_TCM

echo "running kripke for readex-dyn-detect"
mpirun -n 24 ./kripke $KRIPKE_COMMAND
echo "running kripke done"

echo "running readex-dyn-detect"
echo "phase region = $2"
#readex-dyn-detect -t $1 -p $2 -c $3 -v $4 -w $5 scorep-*/profile.cubex
readex-dyn-detect -p "Loop" -t 0.01 scorep-*/profile.cubex
echo
echo "running readex-dyn-detect done" 

#cp -R scorep-* ../RESULTS/
#cp readex_config.xml ../RESULTS/

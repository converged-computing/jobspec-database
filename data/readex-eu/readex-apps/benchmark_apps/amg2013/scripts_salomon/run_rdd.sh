#!/bin/bash

#PBS -A OPEN-12-63
#PBS -q qprod
#PBS -N amg2013_rdd
#PBS -o amg2013_rdd.out
#PBS -e amg2013_rdd.err
#PBS -l select=4:ncpus=24:mpiprocs=2:ompthreads=12:accelerator=true
#PBS -l walltime=01:00:00
###############################################################################

if [ "$PBS_ENVIRONMENT" == "PBS_BATCH" ]; then
  LOC=$PBS_O_WORKDIR
else
  LOC=$(pwd)
fi

cd $LOC/..

TMPFILE=amg2013_rdd.tmp
rm -f $TMPFILE

module purge
source ./readex_env/set_env_rdd.source

#RDD_t=0.001 #1 ms
#RDD_t=0.01 #10 ms
#RDD_t=0.03 #30 ms
RDD_t=0.1 #100 ms
#RDD_t=0.5 #500 ms
RDD_p=main_phase
RDD_c=10
RDD_v=10
RDD_w=10

export SCOREP_PROFILING_FORMAT=cube_tuple
#export SCOREP_METRIC_PAPI=PAPI_TOT_INS,PAPI_L3_TCM

rm -rf scorep-*
rm -f readex_config.xml

mpirun $BIND_TO_SOCKETS ./test/amg2013_rdd -P 2 2 2 -r 40 40 40 2>&1 | tee -a $TMPFILE

readex-dyn-detect -t $RDD_t -p $RDD_p -c $RDD_c -v $RDD_v -w $RDD_w scorep-*/profile.cubex 2>&1 | tee -a $TMPFILE

echo "RDD result = $?"

rm -f $TMPFILE

echo "run RDD done."

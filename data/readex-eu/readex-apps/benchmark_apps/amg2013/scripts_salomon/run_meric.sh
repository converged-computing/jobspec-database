#!/bin/bash

#PBS -A OPEN-12-63
#PBS -q qprod
#PBS -N amg2013_meric
#PBS -o amg2013_meric.out
#PBS -e amg2013_meric.err
#PBS -l select=4:ncpus=24:mpiprocs=2:ompthreads=12:accelerator=true
#PBS -l x86_adapt=true
#PBS -l walltime=24:00:00
###############################################################################

if [ "$PBS_ENVIRONMENT" == "PBS_BATCH" ]; then
  LOC=$PBS_O_WORKDIR
else
  LOC=$(pwd)
fi

cd $LOC/..

TMPFILE=amg2013_meric.tmp
rm -f $TMPFILE

# LOAD MODULES
module purge
source ./readex_env/set_env_meric.source

# SET MERIC OUTPUT
#export MERIC_COUNTERS=perfevent
export MERIC_MODE=1
export MERIC_DEBUG=0
export MERIC_CONTINUAL=1
export MERIC_AGGREGATE=1
export MERIC_DETAILED=0

# FIRST RUN SOMETIMES CAUSES TROUBLE
export MERIC_NUM_THREADS=12
export MERIC_FREQUENCY=25
export MERIC_UNCORE_FREQUENCY=30
export MERIC_OUTPUT_DIR="amg2013_meric_dir"
mpirun $BIND_TO_SOCKETS ./test/amg2013_meric -P 2 2 2 -r 40 40 40
rm -rf $MERIC_OUTPUT_DIR
rm -rf ${MERIC_OUTPUT_DIR}Counters
#exit

echo $MERIC_OUTPUT_DIR

# FOR EACH SETTINGS
for thread in {12..4..-4}
do
  for cpu_freq in {25..13..-4}
  do
    for uncore_freq in {30..14..-4}
    do
      export MERIC_OUTPUT_FILENAME=$thread"_"$cpu_freq"_"$uncore_freq
      export MERIC_NUM_THREADS=$thread
      export MERIC_FREQUENCY=$cpu_freq
      export MERIC_UNCORE_FREQUENCY=$uncore_freq

      echo
      echo TEST $MERIC_OUTPUT_FILENAME | tee -a $TMPFILE
      for repeat in {1..1..1}
      do
        ret=1
        while [ "$ret" -ne 0 ]
        do
          mpirun $BIND_TO_SOCKETS ./test/amg2013_meric -P 2 2 2 -r 40 40 40 2>&1 | tee -a $TMPFILE
          ret=$?
        done
      done 
    done
  done
done

rm -f $TMPFILE


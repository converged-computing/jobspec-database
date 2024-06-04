#!/bin/bash

#PBS -A OPEN-12-63
#PBS -q qprod
#PBS -N amg2013_ptf
#PBS -o amg2013_ptf.out
#PBS -e amg2013_ptf.err
#PBS -l select=5:ncpus=24:mpiprocs=2:ompthreads=12:accelerator=true
#PBS -l x86_adapt=true
#PBS -l walltime=24:00:00
###############################################################################

if [ "$PBS_ENVIRONMENT" == "PBS_BATCH" ]; then
  LOC=$PBS_O_WORKDIR
else
  LOC=$(pwd)
fi

cd $LOC/..

TMPFILE=amg2013_ptf.tmp
rm -f $TMPFILE

module purge
source ./readex_env/set_env_ptf_rapl.source

export PSC_CPU_BIND=$BIND_TO_SOCKETS

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

export SCOREP_SUBSTRATE_PLUGINS=rrl
export SCOREP_RRL_PLUGINS=cpu_freq_plugin,uncore_freq_plugin,OpenMPTP
export SCOREP_RRL_VERBOSE="WARN"

export SCOREP_METRIC_PLUGINS=x86_energy_sync_plugin
export SCOREP_METRIC_X86_ENERGY_SYNC_PLUGIN="*/E"
export SCOREP_METRIC_PLUGINS_SEP=";"
export SCOREP_METRIC_X86_ENERGY_SYNC_PLUGIN_CONNECTION="INBAND"
export SCOREP_METRIC_X86_ENERGY_SYNC_PLUGIN_VERBOSE="WARN"
export SCOREP_METRIC_X86_ENERGY_SYNC_PLUGIN_STATS_TIMEOUT_MS=1000

export SCOREP_MPI_ENABLE_GROUPS=ENV

export OMP_NUM_THREADS=12

PHASE=main_phase

psc_frontend --apprun="./test/amg2013_ptf -P 2 2 2 -r 40 40 40" --mpinumprocs=8 --ompnumthreads=12 --phase=$PHASE --tune=readex_intraphase --config-file=readex_config_ptf.xml --force-localhost --info=2 --selective-info=AutotuneAll,AutotunePlugins 2>&1 | tee -a $TMPFILE

rm -f $TMPFILE


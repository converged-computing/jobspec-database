#!/bin/bash
### set the number of nodes
### set the number of PEs per node
#PBS -l nodes=4:ppn=1:x
### set the wallclock time
#PBS -l walltime=00:29:00
### set the job name
#PBS -N mpiio
### set the job stdout and stderr
#PBS -e $PBS_JOBID.err
#PBS -o $PBS_JOBID.out
### set email notification
#PBS -m bea
#PBS -M gheber@hdfgroup.org
### Get the Darshan log
#PBS -l gres=darshan

export SCRATCH_HOME=/scratch/sciteam/$USER
export DARSHAN_LOGPATH=$SCRATCH_HOME/darshan-logs
export H5PERF_HOME=$SCRATCH_HOME/h5perf-bench
export EXE=$H5PERF_HOME/h5perf
export NUM_PROCS=4
export RUN_DIR=$SCRATCH_HOME/$PBS_JOBID

export PPSIZE=1G
export REPEAT=5
export STRIPE_COUNT=8

export HDF5_PARAPREFIX=$RUN_DIR

mkdir -p $RUN_DIR

lfs setstripe $RUN_DIR --count $STRIPE_COUNT

cd $RUN_DIR

. /opt/modules/default/init/bash
module load darshan cray-hdf5-parallel/1.8.16

aprun -n $NUM_PROCS $EXE -A mpiio -C -i $REPEAT \
  -e $PPSIZE -p $NUM_PROCS -P $NUM_PROCS \
  -B 1G -x 1G -X 1G > out.$PBS_JOBID

cp out.$PBS_JOBID $HOME/jobs/$NUM_PROCS

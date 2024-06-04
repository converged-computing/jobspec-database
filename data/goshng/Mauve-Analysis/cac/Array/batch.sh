#!/bin/bash
#PBS -l walltime=59:00,nodes=2
#PBS -A acs4_0001
#PBS -j oe
#PBS -N Strep
#PBS -q v4
#PBS -t 0-10

cd $PBS_O_WORKDIR

CORESPERNODE=`grep processor /proc/cpuinfo | wc -l`

NODECNT=$(wc -l < "$PBS_NODEFILE")
TASKCNT=`expr $CORESPERNODE \* $NODECNT`
RUNDIR=$PBS_O_WORKDIR
# The job id is something like 613.scheduler.v4linux.
# This deletes everything after the first dot.
JOBNUMBER=${PBS_JOBID%%.*}
echo '============================'
echo $0
echo '============================'

OUTFILE=/tmp/$USER-$PBS_ARRAYID
echo $PBS_ARRAYID > $OUTFILE
cp $OUTFILE $RUNDIR/results/
rm -f $OUTFILE



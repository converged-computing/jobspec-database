#!/bin/bash
#PBS -N cinc_petka
#PBS -q gpu
#PBS -l select=1:ncpus=12:ngpus=1:mem=20gb:gpu_cap=cuda35:scratch_local=30gb
#PBS -l walltime=3:00:00
#PBS -m ae



SINGULARITYFILE=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch\:21.05-py3.SIF
CODEDIR_ORIG=$PBS_O_WORKDIR
DATADIR_ORIG=$PBS_O_WORKDIR/../data
RESULTSDIR=$PBS_O_WORKDIR/..


trap "clean_scratch" TERM EXIT
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR_ORIG/jobs_info.txt
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

cp -R $DATADIR_ORIG $SCRATCHDIR
cp -R $CODEDIR_ORIG $SCRATCHDIR

CODEDIR=$SCRATCHDIR/$(basename "$CODEDIR_ORIG")

SCRIPTNAME=$CODEDIR/run_singularity.sh
chmod +x $SCRIPTNAME
singularity run --nv -B $SCRATCHDIR $SINGULARITYFILE $SCRIPTNAME $CODEDIR $RESULTSDIR

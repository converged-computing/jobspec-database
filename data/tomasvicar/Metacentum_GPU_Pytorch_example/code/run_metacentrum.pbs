#!/bin/bash
#PBS -N gpu_example
#PBS -q gpu
#PBS -l select=1:ncpus=1:ngpus=1:mem=4gb:gpu_cap=cuda35:scratch_local=1gb
#PBS -l walltime=0:10:00
#PBS -m ae

# adjust for your computatinal requirements


# pytorch singularity image path - can be changed to actual version
SINGULARITYFILE=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch\:21.05-py3.SIF
# code to run is expected to be in same folder you call qsub from
CODEDIR_ORIG=$PBS_O_WORKDIR
# data are expected one folder up in folder data
DATADIR_ORIG=$PBS_O_WORKDIR/../data
# results folder provided as input arg for python is selected to be one folder up
RESULTSDIR=$PBS_O_WORKDIR/..

# set autamatic scratch cleaning
trap "clean_scratch" TERM EXIT
# save some info into PBS_O_WORKDIR
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $PBS_O_WORKDIR/jobs_info.txt
# test SCRATCHDIR avaliability (can cause errors sometimes)
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

# copy data and code to scratchdir
cp -R $DATADIR_ORIG $SCRATCHDIR
cp -R $CODEDIR_ORIG $SCRATCHDIR

# create variable with path to code in scratchdir
CODEDIR=$SCRATCHDIR/$(basename "$CODEDIR_ORIG")


SCRIPTNAME=$CODEDIR/run_singularity.sh
# change permisions for running the script
chmod +x $SCRIPTNAME
#run sungularity script
singularity run --nv -B $SCRATCHDIR $SINGULARITYFILE $SCRIPTNAME $CODEDIR $RESULTSDIR
# provide input argumets for singularity script
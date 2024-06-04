#!/bin/bash
#PBS -N retina
#PBS -q gpu
#PBS -l select=1:ncpus=10:ngpus=1:mem=20gb:gpu_cap=cuda35:scratch_local=70gb
#PBS -l walltime=24:00:00
#PBS -m ae

DATADIR=/storage/brno3-cerit/home/tomasvicar/DeepRetinaSegmentation_tmp

trap "clean_scratch" TERM EXIT
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

cp -R $DATADIR $SCRATCHDIR
cd $SCRATCHDIR
chmod +x $SCRATCHDIR/DeepRetinaSegmentation_tmp/DeepRetinaSegmentation/run_singularity.sh
singularity run --nv -B $SCRATCHDIR /cvmfs/singularity.metacentrum.cz/NGC/PyTorch\:21.05-py3.SIF $SCRATCHDIR/DeepRetinaSegmentation_tmp/DeepRetinaSegmentation/run_singularity.sh $DATADIR

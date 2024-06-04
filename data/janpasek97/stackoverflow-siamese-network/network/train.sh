#!/bin/bash
#PBS -N BiDirCodeEncoder2L2Cls_f1
#PBS -l select=1:ncpus=3:ngpus=1:gpu_cap=cuda75:mem=24gb:scratch_local=30gb
#PBS -l walltime=23:00:00
#PBS -q gpu
#PBS -m abe

MODEL_NAME=BiDirCodeEncoder2L2Cls
LOSS=f1_loss
EMBEDDING=custom

DATADIR=/storage/brno3-cerit/home/pasekj/SiameseSearch_BP

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

module add python-3.6.2-gcc
module add python36-modules-gcc
module add tensorflow-2.0.0-gpu-python3

export PYTHONUSERBASE=/storage/brno3-cerit/home/pasekj/.local
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.6/site-packages:$PYTHONPATH
export PYTHONPATH=$SCRATCHDIR/SiameseSearch_BP:$PYTHONPATH

test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

cp -r $DATADIR $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }

cd $SCRATCHDIR/SiameseSearch_BP

python3 network/main.py --model $MODEL_NAME --loss $LOSS --embedding $EMBEDDING > $SCRATCHDIR/stdout.txt || { echo &2 "Calculation ended up erroneously (with a code $?) !!"; exit 3; }

mkdir $DATADIR/out/logs/$MODEL_NAME"_$LOSS"
mkdir $DATADIR/out/checkpoints/$MODEL_NAME"_$LOSS"
mkdir $DATADIR/out/console/$MODEL_NAME"_$LOSS"

cp -r $SCRATCHDIR/SiameseSearch_BP/network/logs $DATADIR/out/logs/$MODEL_NAME"_$LOSS"/ && export CLEAN_SCRATCH=true || CLEAN$
cp -r $SCRATCHDIR/SiameseSearch_BP/network/checkpoints $DATADIR/out/checkpoints/$MODEL_NAME"_$LOSS"/ && export CLEAN_SCRATCH=true || CLEAN$
cp -r $SCRATCHDIR/stdout.txt $DATADIR/out/console/$MODEL_NAME"_$LOSS"/ && export CLEAN_SCRATCH=true || CLEAN$
clean_scratch

#!/bin/bash
#PBS -l walltime=8:00:00
#PBS -l nodes=1:ppn=1
#PBS -r n
#PBS -q @hades
#PBS -j oe
 
module load CUDA
cd $HOME/ift6266h16/projects/cat_dog
NAME=$PBS_JOBID
RESULT_PATH=./result/$NAME
THEANO_FLAGS="device=gpu,floatX=float32,nvcc.fastmath=True,compiledir=$RAMDISK" python train_many.py $NAME > $RESULT_PATH.out 2> $RESULT_PATH.err
rm .$PBS_JOBID.id

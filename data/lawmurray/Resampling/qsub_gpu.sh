#!/bin/sh
#PBS -l walltime=12:00:00,nodes=1:ppn=16:gpus=1,vmem=64gb
#PBS -j oe

source ~/init.sh
cd $PBS_O_WORKDIR
SEED=$PBS_ARRAYID

export KMP_AFFINITY=compact
export CONFIG_OPTS="--seed $SEED --with-cuda"
export OUTPUT_SUFFIX="gpu-$SEED"

./run.sh

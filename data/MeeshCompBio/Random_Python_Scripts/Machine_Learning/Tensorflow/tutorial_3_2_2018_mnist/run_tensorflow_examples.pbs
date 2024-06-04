#!/bin/bash -l
#PBS -l walltime=0:10:00,nodes=1:ppn=24:gpus=2 -q k40
#PBS -l advres=tutorial.469
#PBS -j oe

cd $PBS_O_WORKDIR

module load tensorflow/1.4_gpu

python linear_regression.py >& linear_regression.out

python mnist_softmax.py >& softmax.out

python mnist_deep_net.py >& deep_net.out


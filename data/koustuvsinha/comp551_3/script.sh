#!/bin/bash
#PBS -N Comp551
#PBS -A eeg-641-aa
#PBS -l walltime=300
#PBS -l nodes=1:gpus=2
cd "${PBS_O_WORKDIR}"
module load apps/python/2.7.10 cuda/7.5.18 libs/cuDNN/4
source /home/koustuvs/project/bin/activate
cd "/home/koustuvs/comp551_3"
THEANO_FLAGS='floatX=float32,device=gpu' python runner_lasagne.py

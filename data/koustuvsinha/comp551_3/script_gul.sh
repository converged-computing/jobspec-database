#!/bin/bash
#PBS -N Comp551
#PBS -A eeg-641-aa
#PBS -l walltime=06:00:00
#PBS -l nodes=1:ppn=8:gpus=2
cd "${PBS_O_WORKDIR}"
module load python/2.7.9 CUDA/7.5.18 cuDNN/5.0-ga
source /home/koustuvs/project/bin/activate
cd "/home/koustuvs/comp551_3"
THEANO_FLAGS='floatX=float32,device=gpu' python runner_lasagne.py -e 100 -m 50 > out.log

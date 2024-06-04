#!/bin/bash
# Your job will use 1 node, 28 cores, and 168gb of memory total.
#PBS -q windfall
#PBS -l select=1:ncpus=28:mem=168gb:pcmem=8gb:ngpus=1
### Specify a name for the job
#PBS -N joint_6_rmsprop_250
### Specify the group name
#PBS -W group_list=nlp
### Used if job requires partial node only
#PBS -l place=pack:exclhost
### CPUtime required in hhh:mm:ss.
### Leading 0's can be omitted e.g 48:0:0 sets 48 hours
#PBS -l cput=3360:00:00
### Walltime is how long your job will run
#PBS -l walltime=120:00:00
#PBS -e /home/u25/dongfangxu9/domain_adaptation/log/err_joint6_rmsprop_250
#PBS -o /home/u25/dongfangxu9/domain_adaptation/log/out_joint6_rmsprop_250

module load python/3.6/3.6.5
module load cuda80/neuralnet/6/6.0
module load cuda80/toolkit/8.0.61
source /home/u25/dongfangxu9/df36/bin/activate

cd $PBS_O_WORKDIR

##############singularity run --nv /extra/dongfangxu9/img/keras+theano_gpu-cp35-cuda8-cudnn6.img joint_training_newkeras.py
THEANO_FLAGS='gcc.cxxflags=-march=corei7,base_compiledir=/home/u25/dongfangxu9/.theano/theao1' python3.6 joint_training_newkeras.py


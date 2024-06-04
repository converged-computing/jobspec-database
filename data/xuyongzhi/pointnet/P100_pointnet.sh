#!/bin/bash
#PBS -q gpupascal
#PBS -l walltime=47:00:00
#PBS -l mem=12GB
#PBS -l jobfs=2GB
#PBS -l ngpus=1
#PBS -l ncpus=6
## For licensed software, you have to specify it to get the job running. For unlicensed software, you should also specify it to help us analyse the software usage on our system.
#PBS -l software=tensorflow/1.0.1-python2.7

## The job will be executed from current working directory instead of home.
#PBS -l wd 
#PBS -r y
#PBS -M yongzhi.xu@student.unsw.edu.au
#PBS -m abe

module load tensorflow/1.0.1-python2.7
module use ~access/modules
module load pythonlib/h5py
module list
 
python2.7 ./train.py > P100_pointnet.out

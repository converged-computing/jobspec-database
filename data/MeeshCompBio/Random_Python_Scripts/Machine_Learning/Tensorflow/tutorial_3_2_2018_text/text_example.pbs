#!/bin/bash
#PBS -l walltime=00:10:00 
#PBS -l nodes=1:ppn=24:gpus=2
#PBS -q k40

cd $PBS_O_WORKDIR

module load tensorflow/1.4_gpu
module load python

time ./tensorflow_text.py


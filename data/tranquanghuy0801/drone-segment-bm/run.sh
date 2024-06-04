#!/bin/bash -l

#PBS -N deer_segmentation

#PBS -l walltime=6:00:00
#PBS -l mem=10GB
#PBS -l ncpus=1
#PBS -l ngpus=1
#PBS -l gputype=K40

module load tensorflow/2.1.0-fosscuda-2019b-python-3.7.4

cd /home/n10069275/projects/drone-segment-bm 

source env/bin/activate
 

python3 main_keras.py


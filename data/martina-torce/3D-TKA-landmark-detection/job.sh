#!/bin/bash

#PBS -l select=1:ncpus=4:mem=550gb:ngpus=1
#PBS -l walltime=03:00:00

#PBS -o /rds/general/user/mt1120/home/jobs
#PBS -e /rds/general/user/mt1120/home/jobs

cd $PBS_O_WORKDIR/3D-TKA-landmark-detection

module purge
module load tools/prod
module load TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1

export JOINT_TYPE=hip
python3 train_resnet.py
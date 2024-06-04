#!/bin/bash
#PBS -P CSCI1166
#PBS -N Tune
#PBS -l select=1:ncpus=12:ngpus=1
#PBS -l walltime=10:00:00
#PBS -q gpu_1
#PBS -m abe
#PBS -M xdilo.coetzer@gmail.com
#PBS -o /mnt/lustre/users/xcoetzer/Long_output
#PBS -e /mnt/lustre/users/xcoetzer/Long_error
#PBS

module purge
module add chpc/python/3.6.1_gcc-6.3.0

pushd /mnt/lustre/users/xcoetzer
source cnn_env/bin/activate
python code/Main_tune.py
popd

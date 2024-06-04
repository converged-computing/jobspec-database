#!/bin/bash
#PBS -P Personal
#PBS -q gpu
#PBS -l select=1:ncpus=24:mem=96G:ngpus=1
#PBS -l walltime=00:20:00
#PBS -N cs5242_project
#PBS -j oe

cd ${PBS_O_WORKDIR}
module load tensorflow/1.4
python application.py

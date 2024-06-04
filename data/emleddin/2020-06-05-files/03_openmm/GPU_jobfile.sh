#!/bin/bash
#PBS -q FIXME                       # queue allocation
#PBS -l nodes=FIXME                 # GPU info
#PBS -j oe                          # same output and error file
#PBS -r n                           # Job not re-runnable
#PBS -o OpenMM.err                  # name of error file
#PBS -N 5Y2S_GPU_OpenMM             # name of job for queue

## If you get a segfalt error and you're a TINKER user
## add the .bashrc_blank file to $HOME on Cruntch and 
## uncomment this line.
#source ~/.bashrc_blank

## The specific GPU card to use
## Change "0" to the one you want
export CUDA_VISIBLE_DEVICES=0

## Go to the directory that the job was submitted from
cd $PBS_O_WORKDIR

## Load OpenMM
## K80s use CUDA 8.0, V100s use CUDA 10.0
module load openmm/cuda-8.0
#module load openmm/cuda-10.0

## Run the Python Script and print Terminal output to file
python new_openmm_simulations.py

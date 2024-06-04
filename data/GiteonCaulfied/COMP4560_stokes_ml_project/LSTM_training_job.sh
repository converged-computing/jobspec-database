#!/bin/bash
 
#PBS -l ncpus=240
#PBS -l ngpus=20
#PBS -l mem=800GB
#PBS -l jobfs=5GB
#PBS -q gpuvolta
#PBS -P kr97
#PBS -l walltime=5:00:00
#PBS -l storage=scratch/kr97
#PBS -l wd
  
module load pytorch/1.10.0

python3 LSTM_training.py > /scratch/kr97/$USER/comp4560/$PBS_JOBID.log
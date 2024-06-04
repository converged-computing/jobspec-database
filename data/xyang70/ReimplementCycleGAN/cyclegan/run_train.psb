#!/bin/bash
#PBS -l nodes=01:ppn=16:xk
#PBS -l walltime=48:00:00
#PBS -N final_project_training
#PBS -e $PBS_JOBID.err
#PBS -o $PBS_JOBID.out
#PBS -m bea
#PBS -M hsiuyao2@illinois.edu
cd ~/ReimplementCycleGAN/cyclegan
. /opt/modules/default/init/bash # NEEDED to add module commands to shell
module load python/2.0.1
#module load cudatoolkit
aprun -n 1 -N 1 python Train_notebook.py --batchSize 1 --epochs 75

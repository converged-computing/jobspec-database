#!/bin/bash
#PBS -q debug
#PBS -l walltime=0:15:0,nodes=1:xe:ppn=32,gres=shifter

cd ${PBS_O_WORKDIR}

module load shifter

shifterimg pull luntlab/cs547_project:latest

export OMP_NUM_THREADS=32
aprun -b -n 1 -d 32  -- shifter --image=docker:luntlab/cs547_project:latest --module=mpich,gpu -- python ./src/validate_dataset.py

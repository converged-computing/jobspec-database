#!/bin/bash

#PBS -l select=1:ncpus=30:mpiprocs=4
#PBS -A MHPCC38870258
#PBS -q debug 
#PBS -l walltime=0:30:00
#PBS -j oe
#PBS -N evaluator

export PYTHONWARNINGS="ignore"
cd $WORKDIR/autovalves
module purge
module load pbs tensorflow/1.8.0 spectrum-mpi cuda/9.2
export PYTHONPATH="$WORKDIR/autovalves/simulator/expSun31"
mpiexec -np 4 python learner/cstr_evaluate.py 100

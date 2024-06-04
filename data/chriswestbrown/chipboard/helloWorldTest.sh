#!/bin/bash

#PBS -l select=3:ncpus=5:mpiprocs=5
#PBS -l place=scatter:excl
#PBS -A MHPCC96650N19
#PBS -q debug
#PBS -l walltime=0:01:00
#PBS -o ${WORKDIR}/output.txt
#PBS -e ${WORKDIR}/error.txt
#PBS -N helloWorldTest

cd ${WORKDIR}
module purge
module load anaconda3/5.2.0
source activate mpi4py
module load gcc/5.3.0 cuda/9.2 openmpi/2.1.1/gnu/5.3.0
mpirun python mpiHelloWorld.py

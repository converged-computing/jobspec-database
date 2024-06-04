#!/bin/bash
#PBS -N train_ptb_ns
#PBS -j oe 
#PBS -l select=1:ncpus=1:mpiprocs=1
#PBS -q SINGLE

# mpi

if [ "${PBS_O_WORKDIR}" != "" ];then
  cd ${PBS_O_WORKDIR}
  export PYTHON_EGG_CACHE=/work/$USER/tmp/.python-eggs
  export OMP_NUM_THREADS=1
fi
source ~/.bashrc


date
aprun -n 1 -N 1 python train_ptb_ns.py --test  --out ../../data/machine_learning/train_ptb_ns
date

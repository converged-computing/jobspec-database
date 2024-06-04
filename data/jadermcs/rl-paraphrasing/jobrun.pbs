#!/bin/bash
#PBS -q longa
#PBS -N CTL
#PBS -o ctl.out
#PBS -e ctl.err
#PBS -l walltime=1:00:00
#PBS -l select=1:ncpus=10:ngpus=4:Qlist=Allnodes
source /etc/profile.d/modules.sh
module load python/python3-10
module load cuda/cuda-10.1
source ~/.bashrc #configures your shell to use conda activate
conda activate transformers-env
echo "Inicio: "`date`
mpirun -n 10 intel_mpi_hello_world | sort
sleep 1m
echo "Fim: " `date`
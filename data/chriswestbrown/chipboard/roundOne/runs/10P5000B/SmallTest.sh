#!/bin/bash

#PBS -l select=1:ncpus=10:mpiprocs=10
#PBS -l place=scatter:excl
#PBS -A MHPCC96650N19
#PBS -q debug
#PBS -l walltime=000:30:00
#PBS -j oe
#PBS -N 2Process5000Boards
#PBS -r y
#PBS -m be
#PBS -M m201560@usna.edu,wcbrown@usna.edu,m201362@usna.edu

cd ${HOME}/chipboard/roundOne
module purge
module load anaconda3/5.2.0
source activate mpi4py
module load gcc/5.3.0 openmpi/2.1.1/gnu/5.3.0 tensorflow/1.11.0
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-local-hdd/cuda-9.2/targets/ppc64le-linux/lib/"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/gpfs/home/m201560/chipboard/boost_1_70_0/lib/"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/chipboard/"
export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/gpfs/pkgs/mhpcc/anaconda3-5.0.1/include/python3.6m/"
export PYTHONPATH="$PYTHONPATH:$HOME/chipboard/"
mpirun python 2P5000Boards.py 2P5000_CPU_GPU $PBS_ARRAY_INDEX

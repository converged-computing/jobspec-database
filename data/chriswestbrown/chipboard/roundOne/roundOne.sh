#!/bin/bash

#PBS -l select=32:ncpus=20:mpiprocs=20
#PBS -l place=scatter:excl
#PBS -A MHPCC96650N19
#PBS -q debug
#PBS -l walltime=0:25:00
#PBS -o ${WORKDIR}/roundOneO.txt
#PBS -e ${WORKDIR}/roundTwoE.txt
#PBS -N RoundOneTesting

cd ${HOME}/chipboard/roundOne
module purge
module load anaconda3/5.2.0
source activate mpi4py
module load gcc/5.3.0 cuda/9.2 openmpi/2.1.1/gnu/5.3.0 tensorflow/1.8.0
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-local-hdd/cuda-9.2/targets/ppc64le-linux/lib/"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/gpfs/home/m201560/chipboard/boost_1_70_0/lib/"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/chipboard/"
export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/gpfs/pkgs/mhpcc/anaconda3-5.0.1/include/python3.6m/"
export PYTHONPATH="$PYTHONPATH:$HOME/chipboard/"
echo $(date) > roundOneTime.dt
mpirun -n 2 python roundOneSubmit.py 4FeatureR1
echo $(date) >> roundOneTime.dt

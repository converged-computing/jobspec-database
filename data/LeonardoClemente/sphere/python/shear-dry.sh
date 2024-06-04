#!/bin/sh
#PBS -N shear-dry-hi_mu-hw
#PBS -l nodes=1:ppn=3
#PBS -l walltime=19200:00:00
#PBS -q qfermi
#PBS -M adc@geo.au.dk
#PBS -m abe

# Grendel CUDA
source /com/gcc/4.6.4/load.sh
CUDAPATH=/com/cuda/5.5.22
export PATH=$HOME/bin:$PATH
export PATH=$CUDAPATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDAPATH/lib64:$CUDAPATH/lib:$LD_LIBRARY_PATH

# Manually installed Python modules
export PYTHONPATH=$HOME/.local/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/.local/lib64/python:$PYTHONPATH

# Manually installed Python
#export PATH=/home/adc/.local/bin:$PATH

# Shared Python2.7
PYTHON=/com/python/2.7.6
export PYTHONPATH=$PYTHON/lib:$PYTHONPATH
export PATH=$PYTHON/bin:$PATH

echo "`whoami`@`hostname`"
echo "Start at `date`"

ORIGDIR=/home/adc/code/sphere
#WORKDIR=/scratch/$PBS_JOBID
WORKDIR=$ORIGDIR

#cp -r $ORIGDIR/* $WORKDIR

cd $WORKDIR
nvidia-smi
rm CMakeCache.txt
cmake . && make
cd python
# $ python shear-starter.py <DEVICE> <FLUID> <C_PHI> <C_GRAD_P> <SIGMA_0>
python shear-starter.py 0 0 1.0 1.0 10.0e3 &
python shear-starter.py 1 0 1.0 1.0 20.0e3 &
python shear-starter.py 2 0 1.0 1.0 40.0e3 &
wait

#cp $WORKDIR/output/* $ORIGDIR/output/

echo "End at `date`"

#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l walltime=00:30:00
#PBS -joe
#PBS -mn
#PBS -T flush_cache

module load python/2.7.8/gnu/4.9.2
module unload gnu/4.9.2
module load intel/14.0
module load mkl/11.1
module load openmpi/1.8.3/intel/14.0
module load numpy/1.9.1/intel/14.0/mkl/11.1/python/2.7.8
module load matplotlib/1.4.3/python/2.7.8/numpy/1.9.1
module load scipy/0.15.1/numpy/1.9.1/intel/14.0/mkl/11.1/python/2.7.8

export PYTHONPATH=$PYTHONPATH:$HOME/python/2.7.8/lib/python2.7/site-packages

cd $PBS_O_WORKDIR
python analysis_3D_bootstrap_proper_fkp.py

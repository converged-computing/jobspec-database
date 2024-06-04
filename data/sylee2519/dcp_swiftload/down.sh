#!/bin/sh
#PBS -V
#PBS -N prism
#PBS -q exclusive
#PBS -A etc
#PBS -l select=1:ncpus=68:mpiprocs=24:ompthreads=1 
#PBS -l walltime=04:00:00
#PBS -m abe
#PBS -M sylee2519@naver.com
#PBS -W sandbox=PRIVATE

cd $PBS_O_WORKDIR
module purge
module load craype-x86-skylake gcc/7.2.0 openmpi/3.1.0
module load forge/18.1.2
module load cmake/3.17.4
module load python/3.7
module load tensorflow/1.12.0

python test2.py 1>stdout 2>stderr

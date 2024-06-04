#!/bin/bash -l
#PBS -l select=1:ncpus=2:mpiprocs=1:ngpus=1:ompthreads=2
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -N 256_0NH2_$NUMBER
#PBS -q gpu

source /projects/atomisticsimulations/Manyi/Miniconda3-DeepMD-2.1-test.env

cd $PBS_O_WORKDIR
   lmp -i in.lammps  1>> model_devi.log 2>> model_devi.log 

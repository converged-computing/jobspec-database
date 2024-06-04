#!/bin/bash
#PBS -l select=1:ncpus=64:mpiprocs=64
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N test_qe
#PBS -q cpu

module load amd_apps/gcc-12.2.1/quantum-espresso/7.2-cpu

cd $PBS_O_WORKDIR

mpirun -np 64 pw.x -inp scf.0.in > OUTPUT.out

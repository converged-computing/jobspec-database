#!/bin/bash
#PBS -N dask-test
#PBS -q economy
#PBS -A STDD0004
#PBS -l select=1:ncpus=36:mpiprocs=6:ompthreads=6
#PBS -l walltime=00:20:00
#PBS -j oe
#PBS -m abe

mpirun --np 6 dask-mpi --nthreads 6 --memory-limit 22e9 --interface ib0 --no-scheduler --local-directory /glade/scratch/$USER

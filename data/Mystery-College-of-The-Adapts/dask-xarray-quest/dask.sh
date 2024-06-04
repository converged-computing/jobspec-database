#!/usr/bin/env bash
#PBS -N sample
#PBS -q economy
#PBS -A STDD0004
#PBS -l select=1:ncpus=36:mpiprocs=6:ompthreads=6
#PBS -l walltime=00:20:00
#PBS -j oe

# Qsub template for UCAR CHEYENNE
# Scheduler: PBS

# This writes a scheduler.json file into your home directory
# You can then connect with the following Python code
# >>> from dask.distributed import Client
# >>> client = Client(scheduler_file='~/scheduler.json')

rm -f scheduler.json
mpirun -np 6 dask-mpi --nthreads 6 --memory-limit 22e9 --interface ib0 --local-directory /glade/scratch/$USER

#!/usr/bin/bash -l

#PBS -N dask-mpi-casper-dav
#PBS -q casper
#PBS -A NIOW0001
#PBS -l select=1:ncpus=30:mpiprocs=30:mem=200GB
#PBS -l walltime=00:05:00

echo "Running Dask code with Dask-MPI "
source activate playground  # Activate conda environment
mpirun -np 30 python simulation.py

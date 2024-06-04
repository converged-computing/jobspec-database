#!/bin/bash

#SBATCH -q regular		# qos
#SBATCH -t 1:30:00		# walltime
#SBATCH -N 4			# total nodes requested
#SBATCH --ntasks-per-core=2	# hyperthread
#SBATCH --ntasks-per-node 8	# mpi ranks per node (nprocs*nnodes)
#SBATCH --cpus-per-task 10	# cpus per rank (nthreads)
#SBATCH --hint=multithread

DASK=$HOME/scheduler.json
rm -f $DASK
mpirun -np 32 dask-mpi --scheduler-file $DASK --interface 'ib0' --nthreads=10 --memory-limit='15G' --no-nanny --local-directory=/tmp

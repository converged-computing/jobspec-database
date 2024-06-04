#!/bin/sh

# In this version of the parallel pi estimation, we have hard-coded
# to use 12 threads.  Therefore, we request ppn=12 for 12 cores.
# A modest timeout of 5 minutes is requested, since this computation
# is rather simple.

#PBS -l nodes=1:ppn=12
#PBS -l walltime=0:05:00
#PBS -N pi-parallel
#PBS -j oe

# Change directories to where this script (and our executable) is.
module load intel;
cd "$PBS_O_WORKDIR"
export OMP_NUM_THREADS=12

# Run the computations we want to.
./julia 5000

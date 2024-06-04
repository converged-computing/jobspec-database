#!/bin/bash
################################################################################
#  Submit script for SDSC's Python-based job bundler for XSEDE/SDSC Trestles.
#  This script shows how to bundle applications which are multithreaded with
#  OpenMP.
#
#  Glenn K. Lockwood, San Diego Supercomputer Center             November 2013
################################################################################
#PBS -N bundler.trestles
#PBS -q normal
#PBS -l nodes=2:ppn=32
#PBS -l walltime=1:00:00
#PBS -v Catalina_maxhops=None,QOS=0

TASKS=tasks.OpenMP          # the name of your tasks list

cd $PBS_O_WORKDIR
module load python/2.7.3    # necessary for bundler.py on Gordon

# -npernode 4 means we want to run four tasks per node
# -tpp 8 means each task will use 8 OpenMP threads (see the tasks.OpenMP file)
#    Thus, (2 nodes)*(4 processes per node)*(8 threads per process) = 64 cores,
#    which we requested in the form of nodes=2:ppn=32 above.
ibrun -npernode 4 -tpp 8 \
    /home/diag/opt/mpi4py/mvapich2/intel/1.3.1/lib/python/mpi4py/bin/python-mpi \
    /home/diag/opt/sdsc-user/bundler/bundler.py $TASKS

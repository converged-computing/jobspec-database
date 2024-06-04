#!/bin/bash
# the queue to be used
#PBS -q normal
# specify your project allocation
##PBS -A uic314
# number of nodes and number of processors per node requested
#PBS -l nodes=3:ppn=32
# requested Wall-clock time.
#PBS -l walltime=00:01:00
# name of the standard out file to be "output-file".
#PBS -o output.$PBS_JOBID.txt
# name of the job
#PBS -N Test_Job
#PBS -V
cd $PBS_O_WORKDIR #change to the working directory
PYTHON=/home/diag/opt/mpi4py/mvapich2/intel/1.3.1/lib/python/mpi4py/bin/python-mpi
mpirun_rsh -np 96  -hostfile $PBS_NODEFILE $PYTHON test.py

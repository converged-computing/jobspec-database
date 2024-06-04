#!/bin/bash
################################################################################
#  Submit script for SDSC's Python-based job bundler for XSEDE/SDSC Gordon
#
#  Glenn K. Lockwood, San Diego Supercomputer Center             November 2013
################################################################################
#PBS -N bundler.gordon
#PBS -q normal
#PBS -l nodes=4:ppn=16:native
#PBS -l walltime=1:00:00
#PBS -v Catalina_maxhops=None,QOS=0

TASKS=tasks                 # the name of your tasks list

cd $PBS_O_WORKDIR
module load python/2.7.3    # necessary for bundler.py on Gordon
mpirun_rsh -export \
    -np $PBS_NP \
    -hostfile $PBS_NODEFILE \
    /home/diag/opt/mpi4py/mvapich2/intel/1.3.1/lib/python/mpi4py/bin/python-mpi \
    /home/diag/opt/sdsc-user/bundler/bundler.py $TASKS

#!/bin/bash
#######################################
# Specify nodes, processors per node,
# and maximum running time
#######################################
#PBS -N RUN

#PBS -l select=2:ncpus=24:mpiprocs=24
#PBS -l walltime=72:00:00
#PBS -q general 
#PBS -P ldrd
#######################################
# Enter directory and set PATH
#######################################
#export LD_LIBRARY_PATH=/opt/openmpi-1.6.5/lib:$LD_LIBRARY_PATH
#PATH=$PBS_O_PATH

cd $PBS_O_WORKDIR 
echo "Start: `date`"
#mpirun /home/jinmiao/projects/lammps-16Feb16/src/lmp_mpi < in.W_hyRUN
./run.sh
echo "End: `date`"


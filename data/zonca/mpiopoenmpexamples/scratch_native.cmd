#!/bin/bash
#PBS -q normal
#PBS -N ior_native
#PBS -l nodes=1:ppn=16:native
#PBS -l walltime=00:25:00
#PBS -o ior_scratch_native.out
#PBS -e ior_scratch_native.err
#PBS -V
##PBS -M youremail@xyz.edu
##PBS -m abe
#PBS -A gue998
cd /scratch/$USER/$PBS_JOBID
mpirun_rsh -hostfile $PBS_NODEFILE -np 4 $PBS_O_WORKDIR/IOR-gordon -i 1 -F -b 16g -t 1m -v -v > IOR_native_scratch.log
cp /scratch/$USER/$PBS_JOBID/IOR_native_scratch.log $PBS_O_WORKDIR/

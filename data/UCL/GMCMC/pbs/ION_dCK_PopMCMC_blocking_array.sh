#!/bin/sh
#
# PBS qsub script for Castillo Katz ion channel model using Metropolis Hastings
#
# Specify a maximum execution time of 2 hours
#PBS -l walltime=2:00:00
#
# Ask to run on 2 nodes, each with 8 cores, with 2GB RAM per node
# (i.e. 16 cores in total with 256mb RAM per core)
#PBS -l select=1:ncpus=16:mem=4gb
#

# Load MPI, Intel MKL and HDF5
module load mpi intel-suite hdf5

blocking=( "0,1,2,3;"
"0;1,2,3;"
"0,1,2;3;"
"0,1;2,3;"
"0;1;2,3;"
"0;1,2;3;"
"0,1;2;3;"
"0;1;2;3;"
"0,1,3,2;"
"0;1,3,2;"
"0,1,3;2;"
"0,1;3,2;"
"0;1;3,2;"
"0;1,3;2;"
"0,1;3;2;"
"0;1;3;2;"
"0,2,1,3;"
"0;2,1,3;"
"0,2,1;3;"
"0,2;1,3;"
"0;2;1,3;"
"0;2,1;3;"
"0,2;1;3;"
"0;2;1;3;"
"0,2,3,1;"
"0;2,3,1;"
"0,2,3;1;"
"0,2;3,1;"
"0;2;3,1;"
"0;2,3;1;"
"0,2;3;1;"
"0;2;3;1;"
"0,3,1,2;"
"0;3,1,2;"
"0,3,1;2;"
"0,3;1,2;"
"0;3;1,2;"
"0;3,1;2;"
"0,3;1;2;"
"0;3;1;2;"
"0,3,2,1;"
"0;3,2,1;"
"0,3,2;1;"
"0,3;2,1;"
"0;3;2,1;"
"0;3,2;1;"
"0,3;2;1;"
"0;3;2;1;"
"1,0,2,3;"
"1;0,2,3;"
"1,0,2;3;"
"1,0;2,3;"
"1;0;2,3;"
"1;0,2;3;"
"1,0;2;3;"
"1;0;2;3;"
"1,0,3,2;"
"1;0,3,2;"
"1,0,3;2;"
"1,0;3,2;"
"1;0;3,2;"
"1;0,3;2;"
"1,0;3;2;"
"1;0;3;2;"
"1,2,0,3;"
"1;2,0,3;"
"1,2,0;3;"
"1,2;0,3;"
"1;2;0,3;"
"1;2,0;3;"
"1,2;0;3;"
"1;2;0;3;"
"1,2,3,0;"
"1;2,3,0;"
"1,2,3;0;"
"1,2;3,0;"
"1;2;3,0;"
"1;2,3;0;"
"1,2;3;0;"
"1;2;3;0;"
"1,3,0,2;"
"1;3,0,2;"
"1,3,0;2;"
"1,3;0,2;"
"1;3;0,2;"
"1;3,0;2;"
"1,3;0;2;"
"1;3;0;2;"
"1,3,2,0;"
"1;3,2,0;"
"1,3,2;0;"
"1,3;2,0;"
"1;3;2,0;"
"1;3,2;0;"
"1,3;2;0;"
"1;3;2;0;"
"2,0,1,3;"
"2;0,1,3;"
"2,0,1;3;"
"2,0;1,3;"
"2;0;1,3;"
"2;0,1;3;"
"2,0;1;3;"
"2;0;1;3;"
"2,0,3,1;"
"2;0,3,1;"
"2,0,3;1;"
"2,0;3,1;"
"2;0;3,1;"
"2;0,3;1;"
"2,0;3;1;"
"2;0;3;1;"
"2,1,0,3;"
"2;1,0,3;"
"2,1,0;3;"
"2,1;0,3;"
"2;1;0,3;"
"2;1,0;3;"
"2,1;0;3;"
"2;1;0;3;"
"2,1,3,0;"
"2;1,3,0;"
"2,1,3;0;"
"2,1;3,0;"
"2;1;3,0;"
"2;1,3;0;"
"2,1;3;0;"
"2;1;3;0;"
"2,3,0,1;"
"2;3,0,1;"
"2,3,0;1;"
"2,3;0,1;"
"2;3;0,1;"
"2;3,0;1;"
"2,3;0;1;"
"2;3;0;1;"
"2,3,1,0;"
"2;3,1,0;"
"2,3,1;0;"
"2,3;1,0;"
"2;3;1,0;"
"2;3,1;0;"
"2,3;1;0;"
"2;3;1;0;"
"3,0,1,2;"
"3;0,1,2;"
"3,0,1;2;"
"3,0;1,2;"
"3;0;1,2;"
"3;0,1;2;"
"3,0;1;2;"
"3;0;1;2;"
"3,0,2,1;"
"3;0,2,1;"
"3,0,2;1;"
"3,0;2,1;"
"3;0;2,1;"
"3;0,2;1;"
"3,0;2;1;"
"3;0;2;1;"
"3,1,0,2;"
"3;1,0,2;"
"3,1,0;2;"
"3,1;0,2;"
"3;1;0,2;"
"3;1,0;2;"
"3,1;0;2;"
"3;1;0;2;"
"3,1,2,0;"
"3;1,2,0;"
"3,1,2;0;"
"3,1;2,0;"
"3;1;2,0;"
"3;1,2;0;"
"3,1;2;0;"
"3;1;2;0;"
"3,2,0,1;"
"3;2,0,1;"
"3,2,0;1;"
"3,2;0,1;"
"3;2;0,1;"
"3;2,0;1;"
"3,2;0;1;"
"3;2;0;1;"
"3,2,1,0;"
"3;2,1,0;"
"3,2,1;0;"
"3,2;1,0;"
"3;2;1,0;"
"3;2,1;0;"
"3,2;1;0;"
"3;2;1;0;" );

# Run the program using mpiexec (PBS requires absolute paths to the dataset and result files)
mpiexec ${HOME}/GMCMC/ION_dCK_PopMCMC --num_temperatures=3 --num_burn_in_samples=500 --num_posterior_samples=500 --blocking="fixed:${blocking[PBS_ARRAY_INDEX]}" --dataset=${HOME}/GMCMC/data/ION_dCK_0,5s.h5 ${HOME}/results/ION_dCK_PopMCMC_BurnIn_${PBS_ARRAY_INDEX}.h5 ${HOME}/results/ION_dCK_PopMCMC_Posterior_${PBS_ARRAY_INDEX}.h5
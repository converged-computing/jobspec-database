#!/bin/bash
#PBS -S /bin/bash

#PBS -A GEO111

## job name and output file
#PBS -N go_benchmark
#PBS -j oe
#PBS -o OUTPUT_FILES/job.o
###PBS -q debug

###########################################################

## USER PARAMETERS
##titan: gpu compute nodes have 1 GPU card (K20x) and 16-core (interlagos) CPU

#PBS -l walltime=1:00:00
#PBS -l nodes=96

## mesher runs on CPU cores only
##PBS -l feature=gpu

NPROC=96

###########################################################

#export MPI_DSM_CPULIST=0,1,8,9

# in case multiple mpi processes share same gpu
#export CRAY_CUDA_PROXY=1

# example: with OpenMP
#OMP_NUM_THREADS=8 aprun –n 2 -d 8 -j 1 ./foo

cd $PBS_O_WORKDIR

echo "running simulation: `date`"
echo "directory: `pwd`"
echo
module list
echo

mkdir -p OUTPUT_FILES
mkdir -p DATABASES_MPI

# cleanup
rm -rf OUTPUT_FILES/*
rm -rf DATABASES_MPI/*

# obtain job information
cat $PBS_NODEFILE > OUTPUT_FILES/compute_nodes
echo "$PBS_JOBID" > OUTPUT_FILES/jobid

# stores setup
cp DATA/Par_file OUTPUT_FILES/
cp DATA/CMTSOLUTION OUTPUT_FILES/
cp DATA/STATIONS OUTPUT_FILES/

# runs mesher
echo
echo "running mesher..."
echo `date`
aprun -n $NPROC -N 1 ./bin/xmeshfem3D

# runs simulation
echo
echo "running solver..."
echo `date`
aprun -n $NPROC -N 1 ./bin/xspecfem3D

# cleanup
rm -rf DATABASES_MPI

echo
echo "see results in directory: OUTPUT_FILES/"
echo
echo "done: `date`"



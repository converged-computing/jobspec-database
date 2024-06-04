#!/bin/bash
#PBS -N test_hybrid
#PBS -l nodes=2:ppn=16
#PBS -l walltime=00:01:00
#PBS -o ./output/hybrid_python.out
#PBS -e ./error/hybrid_python.err
module load python-2.7.5
export OMP_NUM_THREADS=8

set -x
cd "$PBS_O_WORKDIR"

# Construct a copy of the hostfile with only 16 entries per node.
# MPI can use this to run 16 tasks on each node.
export TASKS_PER_NODE=2
uniq "$PBS_NODEFILE"|awk -v TASKS_PER_NODE="$TASKS_PER_NODE" '{for(i=0;i<TASKS_PER_NODE;i+=1) print}' > nodefile

#cat nodefile
mpiexec --hostfile nodefile -n 4 -x OMP_NUM_THREADS python hybrid_pure_python.py

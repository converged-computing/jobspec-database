#!/usr/bin/env bash

#SBATCH --partition=instruction
#SBATCH -J task2
#SBATCH -o %x.out -e %x.err
#SBATCH --nodes=2 --cpus-per-task=20 --ntasks-per-node=1
#SBATCH -t 0-00:30:00

module load mpi/openmpi
module load nvidia/cuda/11.8

export OMP_PROC_BIND=spread
export OMP_PLACES=threads

g++ task2_pure_omp.cpp reduce.cpp -Wall -O3 -o task2_pure_omp -fopenmp -fno-tree-vectorize -march=native -fopt-info-vec

n=7
t=1

./task2_pure_omp $n $t



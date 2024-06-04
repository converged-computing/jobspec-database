#!/bin/bash
#SBATCH -o jacobi-np8-4096.out
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --nodelist=node-01

mpirun --mca btl_tcp_if_exclude docker0,lo -np 8 jacobi-mpi 4095 100


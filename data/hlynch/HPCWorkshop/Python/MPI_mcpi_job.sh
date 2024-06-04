#!/bin/bash
#PBS -l nodes=2:ppn=28,walltime=00:5:00
#PBS -N mpi_pi
#PBS -q short

module load shared
module load mvapich2/gcc/64/2.2rc1
module load anaconda/2

cd $HOME/python_hpc
mpirun python pi_MPI.py

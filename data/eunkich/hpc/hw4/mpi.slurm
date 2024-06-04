#!/bin/bash

#SBATCH --job-name=mpi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eunkich@uw.edu

#SBATCH --account=amath
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem=5G
#SBATCH --gpus=0
#SBATCH --time=00-00:10:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --export=all
#SBATCH --output=mpi.csv # where STDOUT goes
#SBATCH --error=mpi.err # where STDERR goes
# Modules to use (optional).
# <e.g., module load singularity>
module load ompi

# Your programs to run.
mpic++ -std=c++14 -o mpi.o mpi.cpp;
echo "func,val,logerr,time,n,n_process";

for i in {1..40}
do
    mpirun -np $i mpi.o $((10 ** 8));
done

for i in {1..6}
do
    mpirun -np 8 mpi.o $((10 ** $i));
done

rm *.o
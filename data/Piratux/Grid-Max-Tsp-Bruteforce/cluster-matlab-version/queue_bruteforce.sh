#!/bin/bash
#SBATCH --partition=main
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=48
#SBATCH --time=0:30:00
module load openmpi
mpiCC -std=c++17 -O2 -o bruteforce-matlab-cluster bruteforce-matlab-cluster.cpp

# -n should match --nodes * --ntasks-per-node (as this will indicate how many processes to spawn)
mpirun -n 960 bruteforce-matlab-cluster

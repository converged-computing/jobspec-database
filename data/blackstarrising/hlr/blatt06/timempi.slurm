#!/bin/bash

#SBATCH --partition=west
#SBATCH --nodes=4
#SBATCH --ntasks=12
#SBATCH --output=timempi.out
. /etc/profile.d/modules.sh
. /etc/profile.d/wr-spack.sh
spack load --dependencies mpi

mpiexec ./timempi

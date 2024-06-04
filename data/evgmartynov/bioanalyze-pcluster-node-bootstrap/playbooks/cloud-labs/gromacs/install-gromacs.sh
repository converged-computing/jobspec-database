#!/usr/bin/env bash

#SBATCH -N 4
#SBATCH -n 4
#SBATCH --partition dev
#SBATCH --constraint m5a4xlarge
#SBATCH --time 1-00:00:00
#SBATCH --exclusive

# First try to install with the binary cache
srun -v -v -N 4 -n 4 spack install \
  -v -y \
  --deprecated \
  --show-log-on-error \
  --no-check-signature \
  --no-checksum \
  gromacs@2022 \
  gromacs@2022 +cuda +mpi \
  gromacs@2022 +cuda ~mpi


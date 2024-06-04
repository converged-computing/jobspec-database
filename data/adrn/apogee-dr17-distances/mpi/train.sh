#!/bin/bash
#SBATCH -J joaquin-train
#SBATCH -o logs/train.o%j
#SBATCH -e logs/train.e%j
#SBATCH -N 4
#SBATCH --ntasks-per-node=64
#SBATCH -t 12:00:00
#SBATCH -p cca
#SBATCH --constraint=rome

source ~/.bash_profile
init_conda

cd /mnt/ceph/users/apricewhelan/projects/apogee-dr17-distances

date

mpirun python3 -m mpi4py.run -rc thread_level='funneled' \
$CONDA_PREFIX/bin/joaquin train -c config.yml -v --mpi

date


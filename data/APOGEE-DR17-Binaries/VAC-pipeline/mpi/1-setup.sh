#!/bin/bash
#SBATCH -J apogee-setup
#SBATCH -o logs/apogee-setup.o%j
#SBATCH -e logs/apogee-setup.e%j
#SBATCH -N 2
#SBATCH -t 04:00:00
#SBATCH -p cca
#SBATCH -C rome

# Relocate and initialize shell
cd /mnt/ceph/users/apricewhelan/projects/apogee-dr17-binaries/vac-pipeline
source hq-config/init.sh
echo $HQ_RUN_PATH

date

mpirun python3 -m mpi4py.run -rc thread_level='funneled' \
$CONDA_PREFIX/bin/hq make_prior_cache -v --mpi

hq make_tasks -v

date

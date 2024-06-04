#!/bin/bash
#SBATCH -J apogee-rerun
#SBATCH -o logs/apogee-rerun.o%j
#SBATCH -e logs/apogee-rerun.e%j
#SBATCH -N 10
#SBATCH --ntasks-per-node=64
#SBATCH -t 36:00:00
#SBATCH -p cca
#SBATCH --constraint=rome

# Relocate and initialize shell
cd /mnt/ceph/users/apricewhelan/projects/apogee-dr17-binaries/vac-pipeline
source hq-config/init.sh
echo $HQ_RUN_PATH

date

mpirun python3 -m mpi4py.run -rc thread_level='funneled' \
$CONDA_PREFIX/bin/hq rerun_thejoker -v --mpi

date


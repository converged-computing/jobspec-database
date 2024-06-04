#!/bin/bash

#SBATCH -n 9
#SBATCH -p debug
#SBATCH -N 3
#SBATCH --ntasks-per-node 3
#SBATCH -J MPICC
#SBATCH -o stdout_%J.txt
#SBATCH -e stderr_%J.txt

export OMP_NUM_THREADS=1
# valid values: hash, ds12, ds21
export PMIX_MCA_gds=hash
# valid values: cma, emulated, none
export OMPI_MCA_btl_vader_single_copy_mechanism=none

srun apptainer exec mpicc.sif mpi-pi
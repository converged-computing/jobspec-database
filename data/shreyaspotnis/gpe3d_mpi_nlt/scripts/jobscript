#!/bin/bash
# MOAB/Torque submission script for SciNet GPC
#
#PBS -l nodes=2:ppn=8,walltime=15:00:00
#PBS -N test

# load modules (must match modules used for compilation)
module purge

# load python modules
module load gcc/4.8.1
module load intel/14.0.1
module load python/2.7.5

# load MPI modules
module unload intel openmpi
module load intel/15.0.2
module load openmpi/intel/1.6.4
module load fftw/3.3.3-intel-openmpi


# DIRECTORY TO RUN - $PBS_O_WORKDIR is directory job was submitted from
cd $PBS_O_WORKDIR

# EXECUTION COMMAND; -np = nodes*ppn
# make sure that gpe3d_mpi_nlt/bin is in $PATH
# mpirun -np 16 gpe3d_mpi_nlt

# If that does not work, use the explicit path instead
mpirun -np 16 ~/code/nltunneling/gpe3d_mpi_nlt/bin/gpe3d_mpi_nlt

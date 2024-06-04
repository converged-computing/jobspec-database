#!/bin/bash
#PBS -l select=1:ncpus=2:mem=1gb
#PBS -l walltime=08:00:00

# NOTE - THIS IS WRITTEN AS A PBS SCRIPT, BUT YOU CAN RUN IT ALL THE SAME AS A SHELL SCRIPT (bash ignores #PBS)

# Configure Environment
module load mpi/intel-2019.8.254
prefix=/rds/general/user/sp2017/home/programs/local
export CPATH=$prefix/include
export INCLUDE=$prefix/include
export LIBRARY_PATH=$prefix/lib
export LD_LIBRARY_PATH=$prefix/lib
export PKG_CONFIG_PATH=$prefix/lib/pkgconfig/
export PLUMED_KERNEL=$prefix/lib/libplumedKernel.so
cd $PBS_O_WORKDIR

# Nice to see what the stats of the node you've been assigned to are
lscpu

# Simulation Parameters
LMP_EXE="/rds/general/user/sp2017/home/programs/lammps/build/lmp"
STEPS=3e7

# Run Simulation
mpirun ${LMP_EXE} -in lammps.in -var outdir . -var tsteps ${STEPS}

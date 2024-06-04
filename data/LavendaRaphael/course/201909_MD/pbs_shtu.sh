#!/bin/bash
#PBS -l nodes=1:ppn=7
#PBS -N ChangeName
#PBS -q spst-lab

set -eo pipefail
module purge
module add compiler/intel/composer_xe_2019.1.053
module add mpi/intelmpi/2019.7
cd $PBS_O_WORKDIR

lmp_bin=~/0example/software/lammps-3Mar20/build/lmp

mpirun ${lmp_bin} < in.melt > out.melt


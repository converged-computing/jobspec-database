#!/bin/bash
#PBS -N coh
#PBS -q workq
#PBS -l select=1:ncpus=16:mpiprocs=16:mem=60gb,walltime=72:00:00
#PBS -j oe

echo "START-------------------------"
qstat -xf $PBS_JOBID

module purge
module add gcc/7.1.0 openmpi/1.8.4 fftw/3.3.4-g481 

cd $PBS_O_WORKDIR

mpiexec -n 16 ~/bin/lammps/lmp_mpi < input.equil

rm -f core.*

#!/bin/bash
#PBS -q cpu
#PBS -l nodes=1:ppn=4
#PBS -l walltime=00:20:00
#PBS -o lammps_log
#PBS -N LAMMPS
#PBS -V

module swap PrgEnv-cray PrgEnv-gnu
module load fftw
module load cudatoolkit
module load lammps

cd $PBS_O_WORKDIR #change to the working directory
aprun -n 4 /N/soft/cle4/lammps/lammps-27Aug13/bin/lmp_xe6 < in.friction

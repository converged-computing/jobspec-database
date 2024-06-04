#!/bin/sh
#SBATCH -J Test_lammps_CPU
#SBATCH -n 4
#SBATCH -t 24:00:00
#SBATCH --constraint=[HSW24|BDW28]

module purge
module load intel/17.2 openmpi/intel/2.0.1 
module load lammps/17Nov16


cd $PWD
time srun lmp_icc_openmpi < in.interface




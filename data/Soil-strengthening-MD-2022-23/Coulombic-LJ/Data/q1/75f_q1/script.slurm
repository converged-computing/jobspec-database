#!/bin/bash
#
#SBATCH --job-name=75f_q1
#SBATCH --output=output.txt
#SBATCH --ntasks-per-node=27
#SBATCH --nodes=8
#SBATCH --time=48:00:00
#SBATCH -p long-28core

module load shared
module load mvapich2/gcc/64/2.2rc1
module load lammps/gcc/3Mar2020-bigbig

cd $HOME/75f_q1

mpirun lmp_bigbig < hydrogel_test.in > output.txt 
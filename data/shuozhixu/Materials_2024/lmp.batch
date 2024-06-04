#!/bin/bash
#
#SBATCH --partition=cm3atou
#SBATCH --job-name=Ag
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --output=Ag.out
#SBATCH --error=Ag.err
#SBATCH --time=168:00:00
#SBATCH --mail-user=youremailaddress@yourinstitution.edu
#SBATCH --mail-type=ALL
#
#################################################

# Defining the executable and scratch directory

cd ${SLURM_SUBMIT_DIR}

rm -f dump.* shear.*

#module load intel/18
#module load LAMMPS/11Aug17-gompi-2017b
#/bin/hostname

module load intel/2022a

# execute lammps
echo "begin lammps"
echo "the job is ${SLURM_JOB_ID}"

mpirun -np $SLURM_NPROCS ~/lammps-mbvo/mylammps/src/lmp_mpi -in lmp.in

echo "lammps out"


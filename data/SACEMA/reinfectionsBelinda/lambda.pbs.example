#!/bin/bash
#PBS -l select=1:ncpus=24:mpiprocs=24
#PBS -P CBBI1106 
#PBS -q smp
#PBS -l walltime=4:00:00
#PBS -o /mnt/lustre/users/blombard/reinfectionsBelinda/oe_files/lamdba2.out
#PBS -e /mnt/lustre/users/blombard/reinfectionsBelinda/oe_files/lambda2.err
#PBS -m be
#PBS -M belindalombard@sun.ac.za
ulimit -s unlimited

# module add chpc/R/3.5.1-gcc7.2.0
module add chpc/BIOMODULES R/4.1.0

cd /mnt/lustre/users/blombard/reinfectionsBelinda

# Run program
make run infections=3

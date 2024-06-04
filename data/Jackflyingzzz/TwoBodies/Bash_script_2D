#!/bin/sh
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=1:mem=60gb
#PBS -m bea
#PBS -N CCS0.10 
#PBS -q pqaero

module load intel-suite
module load mpi
##module load matlab/R2018a

cd /rds/general/user/aeg116/ephemeral/JFM/Coarse/AR2/Re100/Clapping_Constrained/St0.10

./incompact3d > log_file

 


#!/bin/bash
 
#PBS -l walltime=25:00:00,select=1:ncpus=1:mem=32gb
#PBS -N cifarArray
#PBS -A 
#PBS -m abe
#PBS -M 
#PBS -o output_1.txt
#PBS -e error_1.txt
 
################################################################################
 
module load gcc
module load cuda

echo $PBS_NODEFILE;cat $PBS_NODEFILE

cd $PBS_O_WORKDIR

./run_cifar.sh 1

#!/bin/bash
 
#PBS -l walltime=3:00:00,select=1:ncpus=1:mem=128gb
#PBS -J 0-7
#PBS -N femArray0.5
#PBS -A <ACCOUNT>
#PBS -m abe
#PBS -M <EMAIL>
#PBS -o output_^array_index^.txt
#PBS -e error_^array_index^.txt
 
################################################################################
 
module load gcc
module load cuda

echo I am job index: $PBS_ARRAY_INDEX

echo ${PYTHONPATH}
cd $PBS_O_WORKDIR

./run_femnist.sh $(($PBS_ARRAY_INDEX + 1))

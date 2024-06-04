#!/bin/bash

#PBS -u javed
#PBS -N meraz
#PBS -q gpu
#PBS -l select=1:ncpus=20:ngpus=1
#PBS -o out.log
#PBS -V

module load compilers/intel/parallel_studio_xe_2018_update3_cluster_edition

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=0

# SECONDS = 0
echo ---------------------------------------------
echo ----------------JOB Details------------------
echo ---------------------------------------------

echo Job Name:-- $PBS_JOBNAME
echo Job Id  :-- $PBS_JOBID
echo Job Start Time:-- `date`
echo Working directory is $PBS_O_WORKDIR

source activate p6p17c92 

cd /home/javed/megit/codelog	
python pytorch_gpu.py

conda deactivate
echo --------------------------------------------
echo --------------Job Summery-------------------
echo --------------------------------------------

# duration=$SECONDS
# seconds=$((duration % 60))
# min=$((duration / 60))
# minutes=$((min % 60))
# hours=$((min / 60))

echo Job Name:-- $PBS_JOBNAME
echo Job Id  :-- $PBS_JOBID
echo Job End Time:- `date`
# echo Job Duration:- $hours H : $minutes M : $seconds S
	
echo "---------Job End---------"

exit 0;
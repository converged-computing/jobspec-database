#!/bin/bash
#PBS -A cyberlamp
##PBS -l qos=cl_gpu
#PBS -l nodes=1:ppn=1
#PBS -l gpus=1
#PBS -l pmem=12gb
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -M mlp95@psu.edu
#PBS -N sband_gpu

echo "Starting job $PBS_JOBNAME"
date
echo "Job id: $PBS_JOBID"
echo "About to change into $PBS_O_WORKDIR"
cd $PBS_O_WORKDIR
echo "About to start Python"
source activate seti
python process_gbt_data.py /gpfs/group/jtw13/default/gbt_2020/s_band/AGBT20B_999_41 --clobber --gpu
echo "Python exited"
date

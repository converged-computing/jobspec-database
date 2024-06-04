#!/bin/bash
PATH="" #full project path
runfile="pytorch_gpu.py" #train file name
env_name="mptorch17_102" #Conda environment name
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

source activate $env_name 

cd /home/javed/megit/hpc_iiita/pytorch_1gpu	

python $runfile

conda deactivate
echo --------------------------------------------
echo --------------Job Summery-------------------
echo --------------------------------------------

echo Job Name:-- $PBS_JOBNAME
echo Job Id  :-- $PBS_JOBID
echo Job End Time:- `date`
	
echo "---------Job End---------"

exit 0;
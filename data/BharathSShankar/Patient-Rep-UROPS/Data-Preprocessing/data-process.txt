#! /bin/bash
#PBS -P UROPS
#PBS -j oe
#PBS -N Data_Processing
#PBS -q volta_gpu
#PBS -l select=1:ncpus=5:mem=80gb:ngpus=1:mpiprocs=1
#PBS -l walltime=60:00:00
#! bin/bash
cd $PBS_O_WORKDIR;
np=$(cat ${PBS_NODEFILE} | wc -l);
image="/app1/common/singularity-img/3.0.0/pytorch_1.9_cuda11.1_cudnn8_devel_ubuntu20.04.simg"
singularity exec $image bash << EOF > stdout.$PBS_JOBID 2> stderr.$PBS_JOBID

PYTHONPATH=$PYTHONPATH:/home/svu/e0550582/volta_pypkg/lib/python3.8/site-packages
export PYTHONPATH

python split_Data.py


#!/bin/bash
#PBS -P volta_pilot
#PBS -j oe
#PBS -N NLP_Variation_Training_17000
#PBS -q volta_gpu
#PBS -l select=1:ncpus=5:mem=50gb:ngpus=1
#PBS -l walltime=48:00:00

cd $PBS_O_WORKDIR;

image="/app1/common/singularity-img/3.0.0/pytorch_2.0_cuda_12.0_cudnn8-devel_u22.04.sif"

module load singularity
singularity exec -e $image bash << EOF > stdout.$PBS_JOBID 2> stderr.$PBS_JOBID

PYTHONPATH=$PYTHONPATH:/hpctmp/e0543831/virtualenv/lib/python3.10/site-packages
export PYTHONPATH

python GPU_NLP_Model.py
EOF
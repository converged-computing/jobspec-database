#!/bin/bash

#Set up the PBS environment

#PBS -P RDS-FMH-CWGS_2-RW
#PBS -l select=1:ncpus=8:ngpus=2:mem=24gb
#PBS -l walltime=10:00:00

#Load the modules we need
module load singularity
module load cuda/10.0.130

#Change to the directory you qsub-ed this pbs script
cd /scratch/RDS-FMH-CWGS_2-RW/UsydInnovation2019/

#Run the container, call python, and test pytorch
singularity exec --nv container/dtorch.img python script/train_resnet_reg.py resnet_artemis > logs/resnet_artemis.log

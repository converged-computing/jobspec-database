#!/bin/bash 
#PBS -N production1.1 
#PBS -j oe 
#PBS -q gpu
#PBS -l nodes=1:ppn=1:gpus=1:exclusive_process,walltime=25:00:00
#! Mail to user if job aborts
#PBS -m a

input=production1.1
ngpu=1

###############################################################
### You should not have to change anything below this line ####
###############################################################

#! change the working directory (default is home directory)

cd $PBS_O_WORKDIR

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo PBS job ID is $PBS_JOBID
echo This jobs runs on the following GPUs:
echo `cat $PBS_GPUFILE | uniq`

#From GPU file, export visible devices to GPUs
##export CUDA_VISIBLE_DEVICES=`cat $PBS_GPUFILE | awk -F"-gpu" '{ printf A$2;A=","}'`

#Run nvidia-smi on visible GPUs
nvidia-smi
nvidia-smi -q -d COMPUTE
./get_X_GPUs.sh $ngpu 

rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi
 
#! Run the executable
charmm -i $input-"$ngpu"GPUs.inp > $input-"$ngpu"GPUs.out 

#!/bin/sh

#PBS -l nodes=1:ppn=16:nvidia
#PBS -l walltime=00:05:00
#PBS -o output$PBS_JOBID.log
#PBS -N GPU_test_job
#PBS -M drs@bnl.gov
#PBS -j oe

# Print the time and date

date
hostname
 
# Count number of processors

#NODECOUNT=`cat $PBS_NODEFILE | uniq | wc -l`
#PROCCOUNT=`wc -l < $PBS_NODEFILE`
#GPUCOUNT=`cat $PBS_GPUFILE | uniq | wc -l`
 
# $PBS_O_WORKDIR is the directory from where you submitted the job
cd $PBS_O_WORKDIR
 
# load cuda module
#. /usr/local/modules/init/bash

module load cuda
#time ~/CudaSamples/bin/x86_64/linux/release/deviceQuery
#time ~/CudaSamples/bin/x86_64/linux/release/deviceQuery
#~/CudaSamples/bin/x86_64/linux/release/clock

echo ./${program}
echo ./$program

time ./${program} ${args}




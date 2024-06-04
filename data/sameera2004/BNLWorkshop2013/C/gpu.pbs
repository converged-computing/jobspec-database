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
 
# $PBS_O_WORKDIR is the directory from where you submitted the job
cd $PBS_O_WORKDIR
 
module load cuda

echo ./$program ${args}

time ./${program} ${args}




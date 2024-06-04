#!/bin/bash -l

#PBS -N VQS
#PBS -l ncpus=1
#PBS -l ngpus=1
#PBS -q saivt_igpu
#PBS -l gputype=M40
#PBS -l mem=16GB 
#PBS -l walltime=1:00:00
#PBS -W group_list=SAIVT 
cd $PBS_O_WORKDIR 
echo Working directory is $PBS_O_WORKDIR

# Print some other environment information
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo This job runs on the following processors:
NODES=`cat $PBS_NODEFILE`
echo $NODES
echo Queue: $PBS_QUEUE
# Compute the number of processors
NPROCS=`wc -l < $PBS_NODEFILE`
echo This job has allocated $NPROCS nodes
echo -----


# put module loads here 
module load tensorflow/1.4.1-gpu-m40-foss-2016a-python-3.5.1

python demo.py

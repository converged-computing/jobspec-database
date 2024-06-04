#!/bin/bash
###### Job name ######
#PBS -N phjl_analysis
###### Output files ######
#PBS -o output.out
#PBS -e error.err
###### Number of nodes and cores ######
#PBS -l select=1:ncpus=64:mpiprocs=64
###### Specify how many hours do you need ######
#PBS -l walltime=8:00:00
###### Queue name ######
#PBS -q ct64
###### Sandbox ######
#PBS -W sandbox=PRIVATE
###### Sends mail to yourself when the job begins and ends ######
#PBS -M wssu@asiaa.sinica.edu.tw
#PBS -m be
###### Specific the shell types ######
#PBS -S /bin/bash

###### Enter this job's working directory ######
cd $PBS_O_WORKDIR

###### Load modules to setup environment ######
. /etc/profile.d/modules.sh
module purge
module add intel/2020u4 openmpi/4.1.4
module load julia/1.9.4

###### Run your jobs with parameters ######
if [ -n "$PBS_NODEFILE" ]; then
  if [ -f $PBS_NODEFILE ]; then
    NPROCS=`wc -l < $PBS_NODEFILE`
  fi
fi

export JULIA_NUM_THREADS=$NPROCS
echo "Hello HPC!"
echo "Current time: $(date)"
echo "The number of allowed threads is $JULIA_NUM_THREADS"

julia ./Analysis_main.jl disc_00*

echo "End Script!"
echo "Current time: $(date)"

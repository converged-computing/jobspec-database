#!/bin/bash
#PBS -N dadyn
#PBS -j oe
#PBS -l walltime=01:00:00:00
#PBS -l pmem=480mb
#PBS -l nodes=4:ppn=1
# #PBS -M dmerz3@gatech.edu
#PBS -V

# Job proper starts here

# Additional environment setting, if necessary
# export PATH=/some/dir/:$PATH
# run job ________________________
export AMBERHOME=/share/apps/amber11
source /opt/intel/Compiler/11.1/075/bin/iccvars.sh intel64
cd $PBS_O_WORKDIR 

# run job
./go.py

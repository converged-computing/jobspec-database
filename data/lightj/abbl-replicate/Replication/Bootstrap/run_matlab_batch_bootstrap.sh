##################################################
#Copy this file to your home directory and run it with qsub
#"qsub matlab.sh" will start the job
#This script is intended to reserve 12 processors for Matlab worker processes
#In your batch script you can run "parpool('local',12)" to start 12 workers on a node
###################################################

#!/bin/bash
#PBS -N matlabpool
#PBS -l nodes=1:ppn=20,mem=40gb
#PBS -V
#PBS -j oe
#PBS -t 1-50

cd $PBS_O_WORKDIR

#Matlab can clobber it's temporary files if multiple instances are run at once
#Create a job specific temp directory to avoid this
mkdir -p ~/matlabtmp/$PBS_JOBID
export MATLABWORKDIR=~/matlabtmp/$PBS_JOBID

matlab -nodesktop -nosplash  -r "run_bootstrap $PBS_ARRAYID spec_name 20 100 100" >> stdout_${PBS_ARRAYID}.log

#Delete the temporary directory
#rm -rf $MATLABWORKDIR

#!/bin/bash
#

## Name of job.
#PBS -N MultiSubmit

## Set computing requirements.
#PBS -l nodes=1:ppn=8,mem=8,feature=8core
#PBS -l walltime=01:00:00

## Set directory for STDOUT and STDERR files.
#PBS -o /gscratch/stf/jresnick/NewData
## Put STDOUT and STDERR in same file.
#PBS -j oe

## Specify working directory for job.
#PBS -d /gscratch/stf/jresnick/CnerveRepo/Simulation/MATLAB/PDM_Param_JR/ExptFiles
#
module load matlab_2015b
#
cd $PBS_O_INITDIR
#
matlab -nosplash -nodisplay <<SubmitJob

Expt = ITDExpt('ITDSim1',...
	'/gscratch/stf/jresnick/NewData/trimmedExpt.mat',...
	'/gscratch/stf/jresnick/NewData/Threshes.mat','Hyak',0);

exit;
SubmitJob

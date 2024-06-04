#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:01:00
#PBS -j oe
#PBS -M aspen.yoo@nyu.edu
#PBS -l mem=1GB
#PBS -m abe
#PBS -N testing

index=${PBS_ARRAYID}
job=${PBS_JOBID}
walltime_lim=${PBS_WALLTIME}
script_name=${PBS_JOBNAME}
module purge
module load matlab

export MATLABPATH=/home/ay963/matlab-scripts
cat<<EOF | matlab -nodisplay
addpath(genpath('/home/ay963/matlab-scripts'))
addpath(genpath('/home/ay963/wordrecognitionmemory'))


testmodelname = 'REM';
isubj = 1;

filepath = ['wordrecognitionmemory/model/4_fitdata/BPSfits/'];
filename = [filepath 'paramfit_patternbayes_' testmodelname '_subj' num2str(isubj) '.txt']
permission = 'a+';

fileID = fopen(filename,permission)
formatSpec = [repmat('%4.4f ',1,16)  '\r\n'];
fprintf(fileID, formatSpec, repmat($index,1,16)) % save stuff in txt file
fclose(fileID);

EOF



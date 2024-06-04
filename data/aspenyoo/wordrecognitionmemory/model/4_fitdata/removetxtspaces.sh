#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:04:00
#PBS -j oe
#PBS -M aspen.yoo@nyu.edu
#PBS -l mem=2GB
#PBS -m abe
#PBS -N removetxtspaces

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

modelname = 'REM';

for isubj = 1:14;
	removetxtpaces(modelname,isubj')
end

EOF



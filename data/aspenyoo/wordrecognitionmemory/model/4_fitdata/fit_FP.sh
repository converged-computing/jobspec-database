#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -M aspen.yoo@nyu.edu
#PBS -l mem=4GB
#PBS -m abe
#PBS -N FP

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

modelname = 'FP';

if $index > 100;
    blah = num2str($index);
    isubj = str2double(blah(1:end-3));
    binningfn = str2double(blah(end-2));
    joblistnum = str2double(blah(end-1:end)); 
end
switch binningfn
	case 3
		fixparams = [1 6; nan 0];
	case 4
		fixparams = [1 7; nan 0];
end

joblistfile = 'joblist_08012016.txt'; %[]; % will do a different job per person. (if you want same for all, write joblist name)

cluster_wrap2(modelname, binningfn, isubj, joblistnum, joblistfile, fixparams)

EOF



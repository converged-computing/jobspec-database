#!/bin/bash
#SBATCH --partition computeq
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10M
#SBATCH --time=0-00:05:00

# just in case purge the old modules and load Matlab module
module purge
module load matlab

# run matlab program via the run_matlab script
logFile=HPCTemplate.log
mFile=HPCTemplate.m
/public/apps/matlab/R2018a/bin/matlab -nodisplay -nosplash -nodesktop -logfile $logFile -r "run $mFile;quit;"

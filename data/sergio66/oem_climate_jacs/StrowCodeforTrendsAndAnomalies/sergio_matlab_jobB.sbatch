#!/bin/bash

# run this with sbatch --array=1-40 sergio_matlab_jobB.sbatch 

#  Name of the job:
#SBATCH --job-name=RUN_CRIS_ANOM

#  N specifies that 1 job step is to be allocated per instance of
#matlab
#SBATCH -N1

#  This specifies the number of cores per matlab session will be
#available for parallel jobs
#SBATCH --cpus-per-task 1

#  Specify the desired partition develop/batch/prod
#SBATCH --partition=batch
##SBATCH --partition=strow

#SBATCH --qos=short+
#SBATCH --time=00:15:00 
#SBATCH --mem-per-cpu=12000

########################################################################
### check the number of input arguments
if [ $# -gt 0 ]; then
  echo "Your command line contains $# arguments"
elif [ $# -eq 0 ]; then
  echo "Your command line contains no arguments"
fi
########################################################################

if [[ "$1" -eq "" ]]; then
  echo "cmd line arg = DNE, generic run"
  matlab -nodisplay -r "clust_make_spectralrates_sergio_cris; exit"
else
  echo "cmd line arg = DoesExist, use it"
  matlab -nodisplay -r "iCmdLineOption = $1; clust_make_spectralrates_sergio_cris; exit"
fi

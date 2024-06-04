#!/bin/bash
#SBATCH -J matlabjob              # job name
#SBATCH -e jobresults/matlabjob.%j.err       # error file name 
#SBATCH -o jobresults/test%j.out       # output file name 
#SBATCH -N 1                      # request 1 node
#SBATCH -n 16                    # request all 16 cores 
#SBATCH -t 00:02:00               # designate max run time 
#SBATCH -p rtx             # designate queue 

module load matlab
matlab -nodesktop -nodisplay -nosplash < gputest.m

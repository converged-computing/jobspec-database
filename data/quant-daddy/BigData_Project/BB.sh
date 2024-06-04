#!/bin/sh
#BB.sh
#Torque script to run Matlab program

#Torque directives
#PBS -N BB
#PBS -W group_list=yetistats
#PBS -l nodes=1,walltime=70:00:00,mem=80000mb
#PBS -M skk2142@columbia.edu
#PBS -m abe
#PBS -V
#PBS -l nodes=1:ppn=12

#set output and error directories (SSCC example here)
#PBS -o localhost:/vega/stats/users/skk2142/BB
#PBS -e localhost:/vega/stats/users/skk2142/BB


#Command to execute Matlab code
matlab -nosplash -nodisplay -nodesktop -r "experiment" > matoutfile

#Command below is to execute Matlab code for Job Array (Example 4) so that each part writes own output
#matlab -nosplash -nodisplay -nodesktop -r "simPoissGLM($LAMBDA)" > matoutfile.$PBS_ARRAYID

#End of script
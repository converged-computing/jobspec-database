#!/bin/sh
#detection-submit.sh
#Torque script to run Matlab program

#Torque directives
#PBS -N psc-detection
#PBS -W group_list=yetistats
#PBS -l nodes=1:ppn=1,walltime=12:00:00,mem=16000mb
#PBS -M shababo@berkeley.edu
#PBS -m abe
#PBS -V

#set output and error directories (SSCC example here)
#PBS -o localhost:/vega/stats/users/bms2156/log
#PBS -e localhost:/vega/stats/users/bms2156/log

#Command to execute Matlab code
#echo $trace_file
#echo $param_file
#echo $param_ind
#echo $noise_type
#paramfile_base=${param_file##*/}
#paramfile_base=${paramfile_base%.*}
#echo $paramfile_base
matlab -nosplash -nodisplay -nodesktop -r "infer_events_caller('$param_file','$PBS_JOBID')" > matlab-output/matoutfile-${PBS_JOBID}
#-${paramfile_base}

#Command below is to execute Matlab code for Job Array (Example 4) so that each part writes own output
#matlab -nosplash -nodisplay -nodesktop -r "simPoissGLM($LAMBDA)" > matoutfile.$PBS_ARRAYID

#End of script

#!/bin/sh
# altai_rates.sh
# Torque script to extract firing rates from the results of a run of the Altai pipeline

# Torque directives
#PBS -N altai_rates
#PBS -W group_list=yetistats
#PBS -l walltime=1:00:00:00,mem=8192mb
#PBS -M dbp2112@columbia.edu
#PBS -m abe
#PBS -V

#set output and error directories
#PBS -o localhost:/vega/stats/users/dbp2112/ahrens/results/
#PBS -e localhost:/vega/stats/users/dbp2112/ahrens/results/

matlab-R2012b -nosplash -nodisplay -nodesktop -r "t = $PBS_ARRAYID; ratesFromAllFrames" > matoutfile

#End of script

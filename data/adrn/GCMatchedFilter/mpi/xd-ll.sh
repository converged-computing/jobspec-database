#!/bin/sh

# Directives
#PBS -N globber-xd
#PBS -W group_list=yetiastro
#PBS -l nodes=1:ppn=1,walltime=00:15:00,mem=6gb
#PBS -V
#PBS -t 0-1143
#PBS -m n

# Set output and error directories
#PBS -o localhost:/vega/astro/users/amp2217/pbs_output
#PBS -e localhost:/vega/astro/users/amp2217/pbs_output

module load openmpi/1.6.5-no-ib

# print date and time to file
date

cd /vega/astro/users/amp2217/projects/globber/

source activate globber

# New run
python scripts/compute-xd-likelihoods.py -f data/ngc5897/XCov_lg.h5 -x data/ngc5897/xd_trained.pickle -n 2000 -i $PBS_ARRAYID 

date

#End of script

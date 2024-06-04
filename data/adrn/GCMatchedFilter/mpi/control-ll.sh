#!/bin/sh

# Directives
#PBS -N globber-control
#PBS -W group_list=yetiastro
#PBS -l nodes=1:ppn=1,walltime=00:55:00,mem=8gb
#PBS -V
#PBS -t 0-1015
# 1015
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
python scripts/compute-cmd-likelihoods.py -f data/ngc5897/XCov.h5 -n 1000 -i $PBS_ARRAYID --name=control -v --ncompare=10000 --smooth=0.08 -o

date

#End of script

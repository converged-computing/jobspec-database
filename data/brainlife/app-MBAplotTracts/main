#!/bin/bash
#PBS -l nodes=1:ppn=8,vmem=16g,walltime=18:00:00
#PBS -N roitracking
#PBS -V

set -e

echo "plotting tracts"

export SINGULARITYENV_MCR_CACHE_ROOT=`pwd`

time singularity exec -e docker://brainlife/mcr:neurodebian1604-r2017a ./compiled/bsc_MBAplotTracts_BL

#trying to workaround the issue of 
# -- Could not access the MATLAB Runtime component cache. Details: fl:filesystem:NotDirectoryError



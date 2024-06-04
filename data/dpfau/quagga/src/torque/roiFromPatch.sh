#!/bin/sh
# roiFromPatch.sh - runs sparse PCA on a single patch specified by the data
#PBS -W group_list=hpcstats
#PBS -l nodes=1,walltime=10:00:00,mem=2gb
#PBS -V
#PBS -e localhost:/hpc/stats/users/dbp2112/Janelia/logs/stderr.txt
#PBS -o localhost:/hpc/stats/users/dbp2112/Janelia/logs/stdout.txt

/usr/local/bin/matlab-R2012b -r "addpath(genpath('/hpc/stats/users/dbp2112/Janelia/quagga')); roiFromPatch($PBS_ARRAYID,'$config_path');"

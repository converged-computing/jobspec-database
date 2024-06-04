#!/bin/bash

#PBS -l select=1:ncpus=5:mem=80g
#PBS -l maxarray_10=1
#PBS -S /bin/bash
#PBS -M francesco.gualdrini@ieo.it
#PBS -m abe
#PBS -j oe
#PBS -V

# send the R script using singularity shell -B /hpcnfs docker://fgualdr/docker_maude
# print log to files
singularity exec -B /hpcnfs/ docker://fgualdr/docker_maude Rscript /hpcnfs/data/GN2/fgualdrini/Master_batch_scripts/CRISPR_TOOLS_BANCH/Simulations_KERNEL/02_Parser.r &> /hpcnfs/data/GN2/fgualdrini/Master_batch_scripts/CRISPR_TOOLS_BANCH/Simulations_KERNEL/02_Parser.log


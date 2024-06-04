#!/bin/bash

#PBS -P [project code]
#PBS -N pullContainers
#PBS -l walltime=02:00:00
#PBS -l ncpus=1
#PBS -l mem=30GB
#PBS -W umask=022
#PBS -q copyq
#PBS -l wd
#PBS -l storage=scratch/[project code]+gdata/[project code]

module load singularity

# TODO create lint test to check this script and nextflow.config have same containers

# specify singularity cache dir (consistent with gadi.config)
export SINGULARITY_CACHE_DIR=/scratch/$PROJECT/$(whoami)/singularity

# pull containers
singularity pull --dir $SINGULARITY_CACHE_DIR docker://quay.io/biocontainers/gatk4:4.1.4.1--0

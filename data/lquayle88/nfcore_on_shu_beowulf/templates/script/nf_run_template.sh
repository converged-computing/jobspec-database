#!/bin/bash

## slurm scheduler flags

# job name  >>> edit "pipeline_name" for the name of the pipeline you are running e.g. rnaseq <<<
#SBATCH --job-name=pipeline_name

# request resources for the nextflow driver job
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G

# send email >>> edit username <<<
#SBATCH --mail-user=a.surname@institution.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL

# output log file
#SBATCH --output=nextflow.log


## load miniconda module and activate analysis environment

module purge
module load miniconda
source activate nf_env


## define and export variables

# prevent java vm requesting too much memory and killing run
export NXF_OPTS="-Xms1g -Xmx2g"

# path to singularity cache
export NXF_SINGULARITY_CACHEDIR="/path/to/singularity_cache/.singularity"

# path to project directory >>> edit so that this points to your project root directory <<<
NF_PROJECT="/path/to/project/yyyy_mm_dd_project_name"

# project directories  >>> edit the name of the "filestore" e.g. /mnt/parscratch/users <<<
WORK_DIR="${NF_PROJECT}/results/nf_work"
PARAM_DIR="${NF_PROJECT}/src/nf_params"
CONF_DIR="${NF_PROJECT}/src/nf_config"


## run command  >>> edit "pipeline", "version" and "profile" <<<

nextflow run nf-core/pipeline \
-r version \
-profile profile \
-work-dir $WORK_DIR \
-params-file ${PARAM_DIR}/nf-params.json
# -params-file ${PARAM_DIR}/nf-params.json \
# -c ${CONF_DIR}/custom.config

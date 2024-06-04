#!/bin/bash
#PBS -N skynet_1
#PBS -A jvb-000-ag
#PBS -l nodes=1:gpus=1,walltime=00:01:00

module --force purge
PATH=$PATH:/opt/software/singularity-3.0/bin/

# set the working directory to where the job is launched
cd "${PBS_O_WORKDIR}"

# Run you script inside a singularity image
IMAGE=/rap/jvb-000-aa/singularityimages/pytorch.simg

FOLDERS=$RAP,$HOME,$SCRATCH

SINGULARITY_EXEC="singularity exec --nv --bind $FOLDERS $IMAGE"

# start your python script
$SINGULARITY_EXEC python mnist.py

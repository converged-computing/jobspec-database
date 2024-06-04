#!/bin/bash
#PBS -A colosse-users
#PBS -l advres=MILA2019
#PBS -l feature=k80
#PBS -l nodes=1:gpus=1
#PBS -l walltime=00:10:00

module --force purge
PATH=$PATH:/opt/software/singularity-3.0/bin/

# set the working directory to where the job is launched
cd "${PBS_O_WORKDIR}"

# Singularity options
IMAGE=/rap/jvb-000-aa/COURS2019/etudiants/ift6759.simg

RAP=/rap/jvb-000-aa/COURS2019/etudiants/$USER
mkdir -p $RAP
FOLDERS=$RAP,$HOME

SINGULARITY_EXEC="singularity exec --nv --bind $FOLDERS $IMAGE"

# start your python script
$SINGULARITY_EXEC python mnist.py


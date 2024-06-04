#!/bin/bash
#PBS -S /bin/bash
#PBS -N TACHE       # Nom de la tâche
#PBS -A colosse-users  # Identifiant Rap; ID
#PBS -l feature=k80
#PBS -l walltime=6:00:00    # Durée en secondes
#PBS -l nodes=1:gpus=1  # Nombre de noeuds.
# do not execute on login nodes
module --force purge

#XXXPBXXXS -l advres=MILA2019
PATH=$PATH:/opt/software/singularity-3.0/bin/

# set the working directory to where the job is launched
cd "${PBS_O_WORKDIR}"

# Singularity options 
IMAGE=/rap/jvb-000-aa/COURS2019/etudiants/ift6759.simg
RAP=/rap/jvb-000-aa/COURS2019/etudiants/$USER 

mkdir -p $RAP 

s_exec python3 -u -m horoma train --mode TRAIN_ALL --embedding vae --cluster mini_batch_kmeans
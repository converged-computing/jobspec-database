#!/bin/bash

# Nombre de machine ou NODES typiquement=1 sauf
#SBATCH -N 1

# Nombre de processus en general=1 (a mémoire distribues type miprun)
#SBATCH --ntasks=1

#SBATCH --gres=gpu
####SBATCH --gres=gpu:V100-SXM2-32GB:4

# Nom de votre job afficher dans la lise par squeue
#SBATCH --job-name=hazpi_train

# Nom du fichier de sortie et des erreurs avec l'id du job
#####SBATCH --output=res_%j.log
#####SBATCH --error=res_%j.err

#SBATCH --partition=lasti,gpu,gpuv100,gpup6000
####SBATCH --partition=all

# Mail pour etre informe de l'etat de votre job
#SBATCH --mail-type=start,end,fail
#SBATCH --mail-user=gael.de-chalendar@cea.fr

# Temps maximal d'execution du job ci dessous
# d-hh:mm:ss
#SBATCH --time=5-0:00:00

# Taille de la memoire exprime en Mega octets max=190000 ici 50G
#SBATCH --mem=50G

####SBATCH --exclude=node5
####SBATCH --nodelist=node27

#set -o nounset
set -o errexit
set -o pipefail

echo "$0"



# activate environments
source /softs/anaconda/3-5.3.1/bin/activate
source activate hazpi2
module load cuda/11.1
export LD_LIBRARY_PATH=/home/users/gdechalendar/cuda/lib64:${LD_LIBRARY_PATH}
/usr/bin/env python3 --version
if [[ -v SLURM_JOB_ID ]] ; then
    nvidia-smi
    
    # Affiche la (ou les gpus) allouee par Slurm pour ce job
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi
        
echo "Begin on machine: `hostname`"
        
# conda list
        
##############
              
#echo 'Syncing data'
#install -d /scratch/gael/hazpi
#rsync -avz --inplace --delete-delay bergamote2-ib:/scratch_global/gael/concode/data/d_100k_762 /scratch/gael/concode/
#ls /scratch/gael/concode
              
echo 'Script starting'
              
cd /home/users/gdechalendar/Projets/HazPi/HazPi
              
# run script
echo 'Training'
python train_transformer.py -data_path /home/users/jlopez/dataset/medical_articles.xlsx -checkpoint_path /home/users/gdechalendar/Projets/HazPi/checkpoints/  -vocab_save_dir /home/users/gdechalendar/Projets/HazPi/vocab/ -batch_size 12 -epochs 300 -no_filters


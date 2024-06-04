#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1

#module load 2022
#module load Python/3.10.4-GCCcore-11.3.0

DATASET_ID=501
now=$(date)
echo "Hello, this is a ULS job training"
echo "The starting time is $now"
echo "This version is with augmentatation and from NOT! a 901 checkpoint"
echo "Training on 2D"

timestr=$(date +"%Y-%m-%d_%H-%M-%S")
source "/home/ljulius/miniconda3/etc/profile.d/conda.sh"
# conda activate uls

source /home/${USER}/.bashrc
conda activate uls

# ULS env variables
# @julian de vraag is ff of je deze nodig hebt, of dat je andere env variable nodig hebt. Dan moet je die hier ff neerzetten
# @janneke vgm is dit hoe het moet staan
export nnUNet_raw="/projects/0/nwo2021061/uls23/nnUNet_raw"
export nnUNet_preprocessed="/home/ljulius/algorithm/nnunet/nnUNet_preprocessed  "
export nnUNet_results="/home/ljulius/algorithm/nnunet/nnUNet_results"

# env variables for running without docker
# @julian de vraag is ff of je deze nodig hebt, of dat je andere env variable nodig hebt. Dan moet je die hier ff neerzetten
export OUTPUT_DIR="/home/ljulius/data/output/"
export INPUT_DIR="/home/ljulius/data/input/"
export MAIN_DIR="/home/ljulius/"
export TMP_DIR="/scratch-local/ljulius/"

# Dit commando is voor het maken van preproccessed dataset
# Dataset ID is overal het nummer achter DatasetXXX op deze locatie /projects/0/nwo2021061/uls23/nnUNet_raw
# Hier kan je naartoe via https://ondemand.snellius.surf.nl/pun/sys/dashboard
# Hier ook ff inloggen met ljulius en ww
# Dit commando hoeven we maar 1x te draaien omdat we alleen bone images preprocessed gebruiken
# en dit niet onze opslag capaciteit overschreid:
# nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity -pl ExperimentPlannerNoResampling
# -pl ExperimentPlannerNoResampling heeft Hoaze gemaakt om resamplen te regelen op een beter manier.
# de gepreprocessde dataset wordt hier opgeslagen /home/ljulius/algorithm/nnunet/nnUNet_preprocessed dacht ik

# Dataset 501 is bone
# Dit is het train commando, je kan kiezen voor verschillende smaakjes (stukje achter -tr)
# 2d gaat veel sneller dan 3d
# de 0 is vgm de optie die we moeten gebruiken, bij 'all' gaat die trianen en valideren op de hele dataset (heel gek)
# nnUNetv2_train $DATASET_ID 2d 0 -tr nnUNetTrainer_ULS_500_HalfLR
# nnUNetv2_train $DATASET_ID 3d_fullres 0 -tr nnUNetTrainer_ULS_500_HalfLR

# Om te testen of alles werkt kan je dit gebruiken (geeft geen goede output maar
# is meer om te testen of alles werkt:
nnUNetv2_train $DATASET_ID 2d 0 -tr nnUNetTrainer_ULS_50_HalfLR -pretrained_weights algorithm/nnunet/nnUNet_results/Dataset901_Filtered_FSUP/nnUNetTrainer_ULS_500_QuarterLR__nnUNetPlansNoRs__3d_fullres_resenc/fold_all/checkpoint_best.pth --val



# export dataset_id=500
# nnUNetv2_predict -f all -i /projects/0/nwo2021061/uls23/nnUNet_raw/Dataset500_DeepLesion3D/imagesTs -o /home/ljulius/data/output/baseline/Dataset500_DeepLesion3D -d $dataset_id -c nnUNetTrainer_ULS_50_HalfLR

# export dataset_id=502
# nnUNetv2_predict -f all -i /projects/0/nwo2021061/uls23/nnUNet_raw/Dataset502_RadboudumcPancreas/imagesTs -o /home/ljulius/data/output/baseline/Dataset502_RadboudumcPancreas -d $dataset_id -c nnUNetTrainer_ULS_50_HalfLR

# export dataset_id=503
# nnUNetv2_predict -f all -i /projects/0/nwo2021061/uls23/nnUNet_raw/Dataset503_kits21/imagesTs -o /home/ljulius/data/output/baseline/Dataset503_kits21 -d 400 -c nnUNetTrainer_ULS_50_HalfLR

# export dataset_id=504
# nnUNetv2_predict -f all -i /projects/0/nwo2021061/uls23/nnUNet_raw/Dataset504_LIDC-IDRI/imagesTs -o /home/ljulius/data/output/baseline/Dataset504_LIDC-IDRI -d 400 -c nnUNetTrainer_ULS_50_HalfLR

# export dataset_id=505
# nnUNetv2_predict -f all -i /projects/0/nwo2021061/uls23/nnUNet_raw/Dataset505_LiTS/imagesTs -o /home/ljulius/data/output/baseline/Dataset505_LiTS -d 400 -c nnUNetTrainer_ULS_50_HalfLR

# export dataset_id=506
# nnUNetv2_predict -f all -i /projects/0/nwo2021061/uls23/nnUNet_raw/Dataset506_MDSC_Task06_Lung/imagesTs -o /home/ljulius/data/output/baseline/Dataset506_MDSC_Task06_Lung -d 400 -c nnUNetTrainer_ULS_50_HalfLR

# export dataset_id=507
# nnUNetv2_predict -f all -i /projects/0/nwo2021061/uls23/nnUNet_raw/Dataset507_MDSC_Task07_Pancreas/imagesTs -o /home/ljulius/data/output/baseline/Dataset507_MDSC_Task07_Pancreas -d 400 -c nnUNetTrainer_ULS_50_HalfLR

# export dataset_id=508
# nnUNetv2_predict -f all -i /projects/0/nwo2021061/uls23/nnUNet_raw/Dataset508_MDSC_Task10_Colon/imagesTs -o /home/ljulius/data/output/baseline/Dataset508_MDSC_Task10_Colon -d 400 -c nnUNetTrainer_ULS_50_HalfLR

# export dataset_id=509
# nnUNetv2_predict -f all -i /projects/0/nwo2021061/uls23/nnUNet_raw/Dataset509_NIH_LN_ABD/imagesTs -o /home/ljulius/data/output/baseline/Dataset509_NIH_LN_ABD -d 400 -c nnUNetTrainer_ULS_50_HalfLR

# export dataset_id=510
# nnUNetv2_predict -f all -i /projects/0/nwo2021061/uls23/nnUNet_raw/Dataset510_NIH_LN_MED/imagesTs -o /home/ljulius/data/output/baseline/Dataset510_NIH_LN_MED -d 400 -c nnUNetTrainer_ULS_50_HalfLR

# export dataset_id=501
# nnUNetv2_predict -f all -i /home/ljulius/Dataset501_RadboudumcBone/imagesTs -o /home/ljulius/data/output/baseline/Dataset501_RadboudumcBone -d 400 -c nnUNetTrainer_ULS_50_HalfLR


now2=$(date)
echo "Done at $now"


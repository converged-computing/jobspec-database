#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=1

#module load 2022
#module load Python/3.10.4-GCCcore-11.3.0

DATASET_ID=501
now=$(date)
echo "Hello, this is a ULS job running inference on more datasets."
echo "The starting time is $now"

timestr=$(date +"%Y-%m-%d_%H-%M-%S")
source "/home/ljulius/miniconda3/etc/profile.d/conda.sh"
# conda activate uls

source /home/${USER}/.bashrc
conda activate uls

# ULS env variables
# @julian de vraag is ff of je deze nodig hebt, of dat je andere env variable nodig hebt. Dan moet je die hier ff neerzetten
# @janneke vgm is dit hoe het moet staan
export nnUNet_raw="/projects/0/nwo2021061/uls23/nnUNet_raw"
# export nnUNet_preprocessed="/home/ljulius/algorithm/nnunet/nnUNet_preprocessed"
export nnUNet_results="/home/ljulius/algorithm/nnunet/nnUNet_results"
# /home/ljulius/algorithm/nnunet/nnUNet_results/Dataset501_RadboudumcBone/nnUNetTrainerBenchmark_5epochs__nnUNetPlans__2d
# export dataset_id=501

# env variables for running without docker
# @julian de vraag is ff of je deze nodig hebt, of dat je andere env variable nodig hebt. Dan moet je die hier ff neerzetten
# export OUTPUT_DIR="/home/ljulius/data/output/"
# export INPUT_DIR="/home/ljulius/data/input/"
# export MAIN_DIR="/home/ljulius/"
# export TMP_DIR="/scratch-local/ljulius/"

# nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION
# Only specify --save_probabilities if you intend to use ensembling. --save_probabilities will make the command save the predicted probabilities alongside of the predicted segmentation masks requiring a lot of disk space.
# Please select a separate OUTPUT_FOLDER for each configuration!
# Note that per default, inference will be done with all 5 folds from the cross-validation as an ensemble. We very strongly recommend you use all 5 folds. Thus, all 5 folds must have been trained prior to running inference.
# If you wish to make predictions with a single model, train the all fold and specify it in nnUNetv2_predict with -f all

export dataset_id=500
nnUNetv2_predict -f all -i /projects/0/nwo2021061/uls23/nnUNet_raw/Dataset500_DeepLesion3D/imagesTs -o /home/ljulius/data/output/baseline/Dataset500_DeepLesion3D -d 400 -c nnUNetTrainer_ULS_500_QuarterLR__nnUNetPlansNoRs__3d_fullres_resenc

export dataset_id=502
nnUNetv2_predict -f all -i /projects/0/nwo2021061/uls23/nnUNet_raw/Dataset502_RadboudumcPancreas/imagesTs -o /home/ljulius/data/output/baseline/Dataset502_RadboudumcPancreas -d 400 -c nnUNetTrainer_ULS_500_QuarterLR__nnUNetPlansNoRs__3d_fullres_resenc

export dataset_id=503
nnUNetv2_predict -f all -i /projects/0/nwo2021061/uls23/nnUNet_raw/Dataset503_kits21/imagesTs -o /home/ljulius/data/output/baseline/Dataset503_kits21 -d 400 -c nnUNetTrainer_ULS_500_QuarterLR__nnUNetPlansNoRs__3d_fullres_resenc

export dataset_id=504
nnUNetv2_predict -f all -i /projects/0/nwo2021061/uls23/nnUNet_raw/Dataset504_LIDC-IDRI/imagesTs -o /home/ljulius/data/output/baseline/Dataset504_LIDC-IDRI -d 400 -c nnUNetTrainer_ULS_500_QuarterLR__nnUNetPlansNoRs__3d_fullres_resenc

export dataset_id=505
nnUNetv2_predict -f all -i /projects/0/nwo2021061/uls23/nnUNet_raw/Dataset505_LiTS/imagesTs -o /home/ljulius/data/output/baseline/Dataset505_LiTS -d 400 -c nnUNetTrainer_ULS_500_QuarterLR__nnUNetPlansNoRs__3d_fullres_resenc

export dataset_id=506
nnUNetv2_predict -f all -i /projects/0/nwo2021061/uls23/nnUNet_raw/Dataset506_MDSC_Task06_Lung/imagesTs -o /home/ljulius/data/output/baseline/Dataset506_MDSC_Task06_Lung -d 400 -c nnUNetTrainer_ULS_500_QuarterLR__nnUNetPlansNoRs__3d_fullres_resenc

export dataset_id=507
nnUNetv2_predict -f all -i /projects/0/nwo2021061/uls23/nnUNet_raw/Dataset507_MDSC_Task07_Pancreas/imagesTs -o /home/ljulius/data/output/baseline/Dataset507_MDSC_Task07_Pancreas -d 400 -c nnUNetTrainer_ULS_500_QuarterLR__nnUNetPlansNoRs__3d_fullres_resenc

export dataset_id=508
nnUNetv2_predict -f all -i /projects/0/nwo2021061/uls23/nnUNet_raw/Dataset508_MDSC_Task10_Colon/imagesTs -o /home/ljulius/data/output/baseline/Dataset508_MDSC_Task10_Colon -d 400 -c nnUNetTrainer_ULS_500_QuarterLR__nnUNetPlansNoRs__3d_fullres_resenc

export dataset_id=509
nnUNetv2_predict -f all -i /projects/0/nwo2021061/uls23/nnUNet_raw/Dataset509_NIH_LN_ABD/imagesTs -o /home/ljulius/data/output/baseline/Dataset509_NIH_LN_ABD -d 400 -c nnUNetTrainer_ULS_500_QuarterLR__nnUNetPlansNoRs__3d_fullres_resenc

export dataset_id=510
nnUNetv2_predict -f all -i /projects/0/nwo2021061/uls23/nnUNet_raw/Dataset510_NIH_LN_MED/imagesTs -o /home/ljulius/data/output/baseline/Dataset510_NIH_LN_MED -d 400 -c nnUNetTrainer_ULS_500_QuarterLR__nnUNetPlansNoRs__3d_fullres_resenc

export dataset_id=501
nnUNetv2_predict -f all -i /home/ljulius/Dataset501_RadboudumcBone/imagesTs -o /home/ljulius/data/output/baseline/Dataset501_RadboudumcBone -d 400 -c nnUNetTrainer_ULS_500_QuarterLR__nnUNetPlansNoRs__3d_fullres_resenc







now2=$(date)
echo "Done at $now"


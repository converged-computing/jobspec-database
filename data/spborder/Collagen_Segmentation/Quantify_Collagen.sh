#!/bin/sh
#SBATCH --qos=pinaki.sarder-b
#SBATCH --job-name=collagen_quantification
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100gb
#SBATCH --time=01:00:00
#SBATCH --output=collagen_seg_quantification_%j.out

pwd; hostname; date
module load singularity

singularity exec ./collagen_segmentation_latest.sif python3 Collagen_Segmentation/CollagenQuantify.py --test_image_path "/blue/pinaki.sarder/samuelborder/Farzad_Fibrosis/Same_Training_Set_Data/Results/Ensemble_RGB/Testing_Output/" --bf_image_dir "/blue/pinaki.sarder/samuelborder/Farzad_Fibrosis/Same_Training_Set_Data/B/" --f_image_dir "/blue/pinaki.sarder/samuelborder/Farzad_Fibrosis/Same_Training_Set_Data/F/" --output_dir "/blue/pinaki.sarder/samuelborder/Farzad_Fibrosis/Same_Training_Set_Data/Results/Ensemble_RGB/Collagen_Quantification/" --threshold 0.1

date
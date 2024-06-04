#!/bin/bash 

#SBATCH --job-name=multilabel_model
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=model_%A.out
#SBATCH --error=model_%A_%a.err
#SBATCH --partition=stv-gpu
#SBATCH --gres=gpu:4
#SBATCH --gres=gpu:a6000:1




module purge 
module load anaconda3

# Load enviornment containing cellpose
source activate fastai


# Run Segmentation only on RGB Images 
python /hpc/scratch/hdd2/fs541623/Cell_Tox_Assay_080421/FEATURE_EXTRACTION/ModelTest.py 
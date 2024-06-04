#!/bin/bash
#PBS -k o
#PBS -l nodes=1:ppn=16,walltime=24:00:00
#PBS -N conc_${SUBJ}_freesurfer
#PBS -m ae
#PBS -M bacaron@indiana.edu

## Script for performing freesurfer segmentation on anatomical image.
## Inputs are an anatomical image, subject number (subj), and project directory (projdir1). Outputs are all the ouputs from
## freesurfer recon-all.  The aparc+aseg.nii.gz image will be needed for the creation of the white matter mask.
## (github.com/brain-life/pestillilab_projects/Concussion/zero5_make_wm_mask.m). 
## 
## Brad Caron, Indiana University, Pestilli Lab 2017

## The code is optimized and written for Karst at IU 
## https://kb.iu.edu/d/bezu

module load fsl
module load freesurfer/5.3.0
module load matlab/2016a

export SUBJECTS_DIR=/N/dc2/projects/lifebid/Concussion/concussion_test/freesurfer/${SUBJ}
source $FREESURFER_HOME/SetUpFreeSurfer.sh

cd /N/dc2/projects/lifebid/Concussion/concussion_test/freesurfer/${SUBJ}

recon-all -s pilot_hq -i t1_acpc_bet.nii.gz -all -openmp 16

#!/bin/bash

#PBS -k o
#PBS -l nodes=1:ppn=16,walltime=20:00:00
#PBS -N FS_HQ

## Script for running freesurfer reconstruction of t1 anatomical.  WM mask used for mrtrix ensemble due dti_init comptability errors in matlab 2014a, but may not be needed in matlab 2016a.  Script is made for use on IU Karst computer-cluster (PBS designation).  Made by Brent McPherson (2015), adapted and used by Brad Caron (IU Graduate Student, 2016) for microstructure in concussion-prone athletics study.

# load modules
module load fsl/5.0.9
module load freesurfer

# set up Freesurfer
export SUBJECTS_DIR=/N/dc2/projects/lifebid/Concussion/concussion_test/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

subj="1_5"; # subjects in study
projdir1="/N/dc2/projects/lifebid/Concussion/concussion_test"; # path to data directory

for subjects in $subj
	do
		recon-all -s concuss_hq -i $projdir1/$subjects/diffusion_directory/Anatomy/t1_acpc.nii.gz - all
done




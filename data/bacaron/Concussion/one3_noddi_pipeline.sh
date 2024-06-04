#!/bin/bash
#PBS -k o
#PBS -l nodes=1:ppn=16,walltime=5:00:00
#PBS -N noddi_pipeline
#PBS -M bacaron@imail.iu.edu

## NODDI script pipeline
## Performs registration of DWI image to ACPC aligned T1, rotates bvecs, creates camino scheme file, creates b0 brainmask for NODDI, and runs NODDI

## modules
module load fsl/5.0.9
module load camino

## Directories
topdir="/N/dc2/projects/lifebid/Concussion/concussion_real"
noddiDir=$topdir/"NODDI"
subj=$1

## Make directories and copy files over
mkdir -v $NODDIDIR/$SUBJ;
cp -v $TOPDIR/$SUBJ/diffusion_directory/diffusion/data.nii.gz $NODDIDIR/$SUBJ/data_initial.nii.gz;
fslreorient2std $NODDIDIR/$SUBJ/data_initial.nii.gz $NODDIDIR/$SUBJ/data.nii.gz
cp -v $TOPDIR/$SUBJ/diffusion_directory/diffusion/bvecs $NODDIDIR/$SUBJ/bvecs;
cp -v $TOPDIR/1_002/diffusion_directory/diffusion/bvals_normalized $NODDIDIR/$SUBJ/bvals;
cp -v $TOPDIR/$SUBJ/diffusion_data/t1_acpc.nii.gz $NODDIDIR/$SUBJ/;

# Make brainmask for ACPC alignment
cd $NODDIDIR/$SUBJ;
cp -v $TOPDIR/bin/reorgDTI $NODDIDIR/$SUBJ/;
bet $NODDIDIR/$SUBJ/data.nii.gz $NODDIDIR/$SUBJ/nodif_brain_preNODDI -f 0.4 -g 0 -m;

## Align dwi to ACPC space
matlab -nosplash -nodisplay -r "dwiAlignT1NODDI('${SUBJ}')";

## Create b0 brainmask for NODDI
yes | . $NODDIDIR/$SUBJ/reorgDTI $NODDIDIR/$SUBJ/data_acpc.nii.gz;
bet $NODDIDIR/$SUBJ/data_acpc.nii.gz $NODDIDIR/$SUBJ/nodif_brain -f 0.4 -g 0 -m;

## Create Camino scheme file
fsl2scheme -bvecfile $NODDIDIR/$SUBJ/bvecs_rot -bvalfile $NODDIDIR/$SUBJ/bvals -bscale 1 > $NODDIDIR/$SUBJ/bvals.scheme;

## Run NODDI
python $TOPDIR/bin/NODDI_fxn.py NODDI_fxn ${SUBJ};


#!/bin/bash
#PBS -l nodes=1:ppn=8,vmem=16g,walltime=18:00:00
#PBS -N roitracking
#PBS -V

set -x
set -e

if [ ! -f wm_anat.nii.gz ]; then
    [ -z "$FREESURFER_LICENSE" ] && echo "Please set FREESURFER_LICENSE in .bashrc" && exit 1;
    echo $FREESURFER_LICENSE > license.txt
    time singularity exec -e -B `pwd`/license.txt:/usr/local/freesurfer/license.txt docker://brainlife/freesurfer_on_mcr:6.0.2 ./create_wm_mask.sh
fi

parc=$(jq -r .parc config.json)
if [ $parc ]; then
    echo "generate rois"
    time singularity exec -e docker://brainlife/mcr:neurodebian1604-r2017a ./compiled/bsc_GenROIfromPairStringList_BL
else
    rois=$(jq -r .rois config.json)
    ln -s $rois rois #matlab won't be able to read from symlink.. if that's the case do cp -r $rois rois
fi

#roiCount=$(ls $rois/*.mif | wc -l)
#chkVal=0
#if [ $roiCount -eq 0 ]; then
#    echo "generate rois"
#    time singularity exec -e docker://brainlife/mcr:neurodebian1604-r2017a ./compiled/bsc_GenROIfromPairStringList_BL
#fi

echo "tracking"
time singularity exec -e docker://brainlife/mrtrix_on_mcr:1.0 ./trackROI2ROI.sh

echo "generating tracts/"
time singularity exec -e docker://brainlife/mcr:neurodebian1604-r2017a ./compiled/wma_formatForBrainLife_v2

ln -s tracts classification/tracts

echo "generating surfaces/"
time singularity exec -e docker://brainlife/pythonvtk:1.1 ./freesurfer2vtks.py aparc+aseg.nii.gz


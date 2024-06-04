#!/bin/bash
#PBS -l nodes=1:ppn=8,vmem=16g,walltime=4:00:00
#PBS -N endpointGen
#PBS -V

set -e
set -x

[ -z "$FREESURFER_LICENSE" ] && echo "Please set FREESURFER_LICENSE in .bashrc" && exit 1;
echo $FREESURFER_LICENSE > license.txt

if [ ! -f gm_mask.nii.gz ]; then
    time singularity exec -e -B `pwd`/license.txt:/usr/local/freesurfer/license.txt \
        docker://brainlife/freesurfer_on_mcr:6.0.0 ./convertgmmask.sh
fi

#for old wmatools
mkdir -p freesurfer
cp gm_mask.nii.gz freesurfer

#if [ ! -f freesurfer/mri/aparc+aseg.nii.gz ]; then
#    echo "creating gm mask"
#    time singularity exec -e -B `pwd`/license.txt:/usr/local/freesurfer/license.txt docker://brainlife/freesurfer_on_mcr:6.0.0 ./convertgmmask.sh
#fi

time singularity exec -e docker://brainlife/mcr:neurodebian1604-r2017a ./compiled/bsc_classifiedStreamEndpointCortex_BL


#!/bin/bash
#PBS -l nodes=1:ppn=12,vmem=100gb,walltime=96:00:00
#PBS -N pRF
#PBS -V

#export MAXMEM=19000000
fmri=$(jq -r .fmri config.json)
stim=$(jq -r .stim config.json)
fsdir=$(jq -r .output config.json)

[ -z "$FREESURFER_LICENSE" ] && echo "Please set FREESURFER_LICENSE in .bashrc" && exit 1;
echo $FREESURFER_LICENSE > license.txt

if [ ! -f $(jq -r .mask config.json) ]; then
  time singularity exec -e -B `pwd`/license.txt:/usr/local/freesurfer/license.txt docker://brainlife/freesurfer-mini:6.0.1 bash -c "mri_convert ${fsdir}/mri/T1.mgz ./T1.nii.gz && \
							mri_convert ${fsdir}/mri/rh.ribbon.mgz ./rh.ribbon.nii.gz && \
							mri_convert ${fsdir}/mri/lh.ribbon.mgz ./lh.ribbon.nii.gz"
  time singularity exec -e docker://brainlife/fsl:6.0.1 ./create_mask.sh $fmri
fi

mkdir -p prf

time singularity exec -e docker://brainlife/mcr:neurodebian1604-r2017a ./compiled/main

time singularity exec -e -B `pwd`/license.txt:/usr/local/freesurfer/license.txt docker://brainlife/freesurfer-mini:6.0.1 ./create_vtks.sh $fsdir

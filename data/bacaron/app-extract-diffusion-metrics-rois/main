#!/bin/bash
#PBS -l nodes=1:ppn=1,vmem=16gb,walltime=0:20:00
#PBS -N app-penumbra-analysis

inflate=`jq -r '.inflate' config.json`

# inflate ROIs
[[ ${inflate} == 'true' ]] && time singularity exec -e docker://brainlife/mcr:neurodebian1604-r2017a ./compiled/inflateROIs

# move ROI to size and dimensions of dwi
echo $FREESURFER_LICENSE > license.txt
time singularity exec -e -B `pwd`/license.txt:/usr/local/freesurfer/license.txt docker://brainlife/freesurfer:6.0.0 ./vol2vol_freesurfer.sh

# compute ROI metrics
time singularity exec -e docker://brainlife/fsl:5.0.9 ./fslanalysis.sh

mv *.nii.gz ./raw/
rm -rf *.txt*
exit 0

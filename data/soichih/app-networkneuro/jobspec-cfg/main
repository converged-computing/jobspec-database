#!/bin/bash
#PBS -l nodes=1:ppn=8:vmem=32gb
#PBS -l walltime=10:00:00
#PBS -N networkneuro

set -e
set -x

export fsdir=`jq -r '.fsdir' config.json`

echo "converting aparc aseg"
[ -z "$FREESURFER_LICENSE" ] && echo "Please set FREESURFER_LICENSE in .bashrc" && exit 1;
echo $FREESURFER_LICENSE > license.txt
time singularity exec -e -B `pwd`/license.txt:/usr/local/freesurfer/license.txt docker://brainlife/freesurfer:6.0.0 mri_convert $fsdir/mri/aparc+aseg.mgz --out_orientation RAS aparc+aseg.nii.gz

echo "running networkneuro"
time singularity exec -e docker://brainlife/mcr:neurodebian1604-r2017a ./compiled/main
if [ ! -s output/pconn.mat ];
then
    echo "output missing"
    exit 1
fi

echo "generating surfaces/ vtk from aparc+aseg.mgz"
time singularity exec -e docker://brainlife/docker-pythonvtk ./freesurfer2vtks.py 


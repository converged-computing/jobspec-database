#!/bin/bash
#PBS -l nodes=1:ppn=8,vmem=32gb,walltime=6:00:00
#PBS -N wmaSeg
#PBS -V

rm -rf tracts

mkdir -p freesurfer/mri
echo "converting aparc.a2009+aseg file to nifti"
module load singularity 2> /dev/null
time singularity exec -e docker://brainlife/freesurfer:6.0.0 bash -c "echo $FREESURFER_LICENSE > /usr/local/freesurfer/license.txt && ./convertaseg.sh"

#TODO..

singularity exec -e docker://brainlife/mcr:neurodebian1604-r2017a ./compiled/main


if [ ! -s output.mat ];
then
	echo "output missing"
	exit 1
fi

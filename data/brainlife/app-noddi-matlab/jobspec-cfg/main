#!/bin/bash
#PBS -l nodes=1:ppn=16,vmem=64gb,walltime=32:00:00
#PBS -N noddi_matlab

set -e

# create brainmask (if necessary) and unzip files
time singularity exec -e docker://brainlife/fsl:5.0.9 ./brainmask.sh

# run noddi_matlab
time singularity exec -e docker://brainlife/mcr:neurodebian1604-r2017a ./compiled/noddi_matlab

# cleanup and error check
if [ -f noddi_fit_ficvf.nii ];then
	mkdir noddi;
	mv noddi_fit_ficvf.nii ./noddi/icvf.nii;
	mv noddi_fit_fiso.nii ./noddi/isovf.nii;
	mv noddi_fit_odi.nii ./noddi/od.nii;
	gzip ./noddi/icvf.nii;
	gzip ./noddi/isovf.nii;
	gzip ./noddi/od.nii;
#	gzip mask.nii;
#	rm -rf *noddi_fit*;
else
	echo "output missing"
	exit 1
fi

#!/bin/bash
#PBS -l nodes=1:ppn=1,vmem=32gb,walltime=0:30:00
#PBS -N app-denoise-tensorflow

time singularity exec -e docker://brainlife/tensorflow-gpu:1.14 ./denoise_pretrained.py

# clean up
mkdir -p dwi
bvals=`jq -r '.bvals' config.json`
bvecs=`jq -r '.bvecs' config.json`
cp ${bvals} ./dwi/dwi.bvals
cp ${bvecs} ./dwi/dwi.bvecs
mv dwi.nii.gz ./dwi/

if [ -f ./dwi/dwi.nii.gz ]; then
  echo "denoising completed"
  exit 0;
else
	echo "denoising failed"
	exit 1;
fi

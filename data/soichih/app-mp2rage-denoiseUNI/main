#!/bin/bash

#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:20:00
#PBS -l vmem=30gb

singularity exec -e docker://brainlife/mcr:r2019a ./compiled/mp2rage_denoiseUNI.m

matlab -nosplash -nodisplay -r mp2rage_denoiseUNI.m
if [ ! -f ./output/MP2RAGE_denoised.nii.gz ]; then
       echo "mp2rage_denoiseUNI.m didn't produce MP2RAGE_denoised.nii.gz"
       exit 1
fi

echo "all done"

#!/bin/bash
#PBS -l nodes=1:ppn=4,vmem=28gb,walltime=00:20:00
#PBS -N VoxelronProcess

set -e
set -x

echo "processing data"docker://brainlife/mcr:r2019a
singularity exec -e docker://brainlife/mcr:r2019a ./compiled/main

#cp labels.json classification
#cp aparc.a2009s+aseg.nii.gz classification/parc.nii.gz

echo "all done"

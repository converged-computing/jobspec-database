#!/bin/bash
#PBS -l nodes=1:ppn=16
#PBS -l walltime=02:00:00

set -x
set -e

input=$(jq -r .t1 config.json)

#pick cpu v.s. nvidia gpu
device=cpu
nvopts=""
if hash nvidia-smi; then
    #TODO - pick an appropriate device instead of 0
    device=0 
    nvopts="--nv"
fi

time singularity run $nvopts -e docker://anibalsolon/hd-bet:v0.0.1 hd-bet \
    -i $input \
    -o masked.nii.gz \
    -device $device \
    -mode fast \
    -tta 0

mkdir -p mask
cp masked_mask.nii.gz mask/t1.nii.gz

mkdir -p output
cp masked.nii.gz output/t1.nii.gz

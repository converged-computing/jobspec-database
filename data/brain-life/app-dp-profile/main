#!/bin/bash
#PBS -l nodes=1:ppn=6
#PBS -l vmem=31gb
#PBS -l walltime=18:00:00

rm -rf results output

MAXMEM=16000000 singularity exec docker://brainlife/mcr:neurodebian1604-r2017a ./compiled/compute_profiles
if [ ! -f output/dt6.mat ]; then
    echo "failed to produce output"
    exit 1
fi


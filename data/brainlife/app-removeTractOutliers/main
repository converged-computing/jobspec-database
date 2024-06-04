#!/bin/bash
#PBS -l nodes=1:ppn=8,vmem=28gb,walltime=10:00:00
#PBS -N wmaSeg
#PBS -V

set -e
set -x

echo "running outlier removal"
singularity exec -e docker://brainlife/mcr:r2019a ./compiled/main

if [ ! -f "classification/classification.mat" ]; then
	echo "output missing"
	exit 1
fi

classification=$(jq .classification -r config.json)
surface_dir=$(dirname $classification)/surfaces
if [ -d $surface_dir ]; then
    echo "copying surfaces directory from the wmc input"
    cp -r $surface_dir classification
fi

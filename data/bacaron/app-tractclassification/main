#!/bin/bash
#PBS -l nodes=1:ppn=8,vmem=58gb,walltime=5:00:00
#PBS -N tractclassification
#PBS -V

rm -rf tracts

chmod -R +rwx dtiinit
rm -rf dtiinit

set -e
set -x

#need to copy dtiinit input to cwd.. because AFQ tries to write stuff to input
#https://github.com/yeatmanlab/AFQ/issues/35
dtiinit_dir=$(jq -r .dtiinit config.json)
cp -r $dtiinit_dir dtiinit
chmod -R +rwx dtiinit

#need to copy dtiinit input to cwd.. because AFQ tries to write stuff to input
#https://github.com/yeatmanlab/AFQ/issues/35
rm -rf dtiinit
cp -r $(jq -r .dtiinit config.json) dtiinit

singularity exec -e docker://brainlife/mcr:r2019a ./compiled/main

if [ ! -s classification.mat ];
then
	echo "output missing"
	exit 1
fi

mkdir -p wmc
mv classification.mat wmc
mv tracts wmc

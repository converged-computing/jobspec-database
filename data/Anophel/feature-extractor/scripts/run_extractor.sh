#!/bin/bash
#PBS -N Features_extraction
#PBS -q gpu
#PBS -l select=1:ncpus=1:mem=20gb:scratch_local=40gb:ngpus=1:gpu_cap=cuda60
#PBS -l walltime=2:00:00

DATADIR=/storage/plzen1/home/anopheles

echo Scratch dir: $SCRATCHDIR
echo Working dir: `pwd`
ls -l $DATADIR/scripts/run_extractor_singularity.sh

BASEDIR=#VID#

echo Vid: $BASEDIR

cd $SCRATCHDIR

mkdir ./$BASEDIR
cp $DATADIR/img/$BASEDIR*/* ./$BASEDIR

find ./$BASEDIR -type f > imagelist.txt

cp -r $DATADIR/feature-extractor ./feature-extractor

mkdir output

#cp $DATADIR/scripts/run_extractor_singularity.sh ~

singularity run --bind $SCRATCHDIR:/scratch --nv /cvmfs/singularity.metacentrum.cz/NGC/TensorFlow\:21.12-tf2-py3.SIF $DATADIR/scripts/run_extractor_singularity.sh

mkdir $DATADIR/extracted_features/$BASEDIR

cp ./imagelist.txt ./output
mv ./output/* $DATADIR/extracted_features/$BASEDIR

rm -rf *

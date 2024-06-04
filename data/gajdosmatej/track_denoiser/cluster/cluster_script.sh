#!/bin/bash
#PBS -N batch_job
#PBS -l select=1:mem=32gb:scratch_local=20gb:ngpus=1:gpu_cap=cuda60
#PBS -l walltime=24:00:00
#PBS -q gpu

DATADIR=/storage/plzen1/home/gajdoma6

rm -fr $DATADIR/models/*
rm -fr $DATADIR/histories/*

test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

cp -r $DATADIR/cluster/run_several_architectures.py $DATADIR/cluster/architectures_list.py $DATADIR/cluster/classes.py $SCRATCHDIR
cd $SCRATCHDIR

mkdir ./models
mkdir ./histories

singularity exec -B $SCRATCHDIR:/scratchdir -B $DATADIR/data:/data --nv /cvmfs/singularity.metacentrum.cz/NGC/TensorFlow\:22.12-tf2-py3.SIF python /scratchdir/run_several_architectures.py

mv ./models $DATADIR/
mv ./histories $DATADIR/

rm -fr $DATADIR/cluster/*

clean_scratch

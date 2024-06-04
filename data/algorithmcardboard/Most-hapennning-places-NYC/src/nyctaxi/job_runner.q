#!/bin/bash
#PBS -l nodes=1:ppn=12
#PBS -l walltime=8:00:00
#PBS -l mem=64GB
#PBS -N zipCode_adder
#PBS -M ajr619@nyu.edu
#PBS -m e
#PBS -j oe

module purge

SRCDIR=$HOME/project/Most-hapennning-places-NYC/
RUNDIR=$SCRATCH/Most-hapennning-places-NYC/run-${PBS_JOBID/.*}
mkdir -p $RUNDIR

cd $PBS_O_WORKDIR
cp -R $SRCDIR/* $RUNDIR

cd $RUNDIR

module load virtualenv/12.1.1;
module load scipy/intel/0.16.0
module load geos/intel/3.4.2

virtualenv .venv

source .venv/bin/activate;

pip install shapely
pip install geopy

cd src/nyctaxi

python zip_adder.py

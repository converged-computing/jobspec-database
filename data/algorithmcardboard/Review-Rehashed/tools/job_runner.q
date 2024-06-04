#!/bin/bash
#PBS -l nodes=1:ppn=4
#PBS -l walltime=24:00:00
#PBS -l mem=8GB
#PBS -N amazon_sitemap_downloader
#PBS -M ajr619@nyu.edu
#PBS -m e
#PBS -j oe

module purge

SRCDIR=$HOME/workspace/Review-Rehashed/tools/
RUNDIR=$SCRATCH/Review-Rehashed/run-${PBS_JOBID/.*}
mkdir -p $RUNDIR

cd $PBS_O_WORKDIR
cp -R $SRCDIR/* $RUNDIR

cd $RUNDIR

module load virtualenv/12.1.1;
module load scipy/intel/0.16.0
module load geos/intel/3.4.2

virtualenv .venv

source .venv/bin/activate;

pip install geopy

python fetch_all_sitemaps.py


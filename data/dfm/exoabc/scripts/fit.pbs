#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=24:00:00
#PBS -l mem=4GB
#PBS -N exoabc-array
#PBS -M danfm@nyu.edu
#PBS -j oe

module purge
export PATH="$HOME/miniconda3/bin:$PATH"
export OMP_NUM_THREADS=1
export EXOABC_DATA=$SCRATCH/exoabc/data

export SRCDIR=$HOME/projects/exoabc
export PATH="$SRCDIR:$PATH"

export RUNDIR=$PBS_O_WORKDIR
mkdir -p $RUNDIR

cd $RUNDIR
python $SRCDIR/scripts/exoabc-fit ${PBS_ARRAYID} --config config.ini --verbose


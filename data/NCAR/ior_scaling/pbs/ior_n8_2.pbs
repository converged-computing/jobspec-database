#!/bin/bash
#PBS -N ior_n8
#PBS -A NIOW0001
#PBS -l walltime=02:00:00
#PBS -q regular
#PBS -j oe
#PBS -l select=8:ncpus=1:mpiprocs=1

export TMPDIR=/glade/scratch/$USER/tmp
mkdir -p $TMPDIR

module restore ncar-ior
export LD_LIBRARY_PATH=/glade/work/kpaul/software/boost/lib:$LD_LIBRARY_PATH

./scripts/runtest.sh 8 4 5 0 5

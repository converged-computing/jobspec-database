#!/bin/bash
#PBS -l walltime=00:30:00
#PBS -l nodes=1
#PBS -A STF007

# Do not load Paraview here, it will clobber python with it's own built in python
# Instead load it in runX.sh
module switch PrgEnv-pgi PrgEnv-gnu
module load wraprun
module load GPU-render

cd $PBS_O_WORKDIR

wraprun -n 1 -b --w-no-ld-pre ./runX.sh app.py

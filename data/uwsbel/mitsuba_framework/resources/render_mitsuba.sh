#!/bin/bash

#PBS -N rectangle_render
#PBS -l nodes=1:ppn=8:amd
#PBS -t 0-583
#PBS -j oe -o /dev/null
cd $PBS_O_WORKDIR
mkdir -p /dev/shm/$PBS_JOBID/
../../mitsuba_converters/converter_fluid $PBS_ARRAYID /dev/shm/$PBS_JOBID/
mitsuba -D frame=/dev/shm/$PBS_JOBID/$PBS_ARRAYID -o output/$PBS_ARRAYID.png -p 8 scene.xml
rm -rf /dev/shm/$PBS_JOBID

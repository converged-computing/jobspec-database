#!/bin/bash
#PBS -l nodes=1:ppn=1,vmem=8g,walltime=0:05:00
#PBS -N trk2tck
#PBS -V

module load singularity 2> /dev/null

set -e
singularity exec -e docker://brainlife/dipy:0.14 ./main.py



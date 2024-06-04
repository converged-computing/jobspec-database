#!/bin/bash
#PBS -N xxjobnamexx
#PBS -j oe
#PBS -l walltime=27:00
#PBS -l pmem=400mb
#PBS -l nodes=1:ppn=1
#PBS -V

# job properties
cd $PBS_O_WORKDIR

NUM=xxnumxx

# run job
./$NUM-continue.py $NUM

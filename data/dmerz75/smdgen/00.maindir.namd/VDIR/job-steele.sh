#!/bin/bash
#PBS -q xxqueuexx
#PBS -N xxjobnamexx
#PBS -l pmem=220mb
#PBS -l xxwalltimexx
#PBS -l xxnodesxx
#PBS -j oe
#PBS -V

# job_________________________
module load namd/2.9b2-tcp
module load python/2.7.2

cd $PBS_O_WORKDIR

# run job
./go.py

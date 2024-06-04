#!/bin/bash
#PBS -l nodes=1:ppn=16:xk
#PBS -l walltime=00:05:00

module load craype-accel-nvidia35

cd $PBS_O_WORKDIR

# Run the OpenACC executable on the XK node with the output suppressed and a
# 10K by 10K world:
aprun -n 1 ./diffusion-openacc.exe -q -w 10000 -h 10000

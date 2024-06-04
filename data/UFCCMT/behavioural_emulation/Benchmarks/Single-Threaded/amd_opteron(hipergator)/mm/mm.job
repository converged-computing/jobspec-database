#!/bin/sh
#
#PBS -r n
#PBS -N mm
#PBS -o mm_log.out
#PBS -m e
#PBS -M kevincheng88@gmail.com
#PBS -l nodes=1:ppn=1
#PBS -l pmem=800mb
#PBS -l walltime=06:00:00

cd $PBS_O_WORKDIR

cat /proc/cpuinfo

module load python/2.7.8

python example-a.py
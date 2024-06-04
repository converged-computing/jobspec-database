#!/bin/bash
#PBS -N clumping_gtex
#PBS -o logs/${PBS_JOBNAME}.o${PBS_JOBID}
#PBS -e logs/${PBS_JOBNAME}.e${PBS_JOBID}
#PBS -l walltime=4:00:00
#PBS -l nodes=1:ppn=1
#PBS -l mem=30G
#PBS -l vmem=30G
#PBS -t 1-87%40

#clumping needs 60 GB RAM
#mr works with 30 gb ram

source .bashrc
cd /scratch/richards/julian.willett/7*
Rscript Main_Revised_For_Time.R ${PBS_ARRAYID} > logs/${PBS_JOBID}.tmp


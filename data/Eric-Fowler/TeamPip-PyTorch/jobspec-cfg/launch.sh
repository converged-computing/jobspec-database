#!/bin/bash
TIME=`date +\%s`
mkdir models/$TIME
cp Script.py models/$TIME/
cp job.slurm models/$TIME/
sbatch --output=models/$TIME/log.out --export=TIME=$TIME job.slurm

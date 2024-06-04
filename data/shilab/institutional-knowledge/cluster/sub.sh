#!/bin/bash
#######################################################################
# ======= PBS OPTIONS ======= 
### Specify queue to run - mamba, Cobra, or copperhead
#PBS -q mamba
### Set the job name
#PBS -N job-name
### Specify the # of cpus for your job.
#PBS -l nodes=1:ppn=1
### Adjust walltime below (default walltime = 7 days, or 168 hours)
### if you require > 7 days, INCREASE to estimated # hours needed
### if you DON'T require 7 days DECREASE to estimated # hours needed
### (hint: jobs with smaller walltime value tend to run sooner)
#PBS -l walltime=168:00:00
### pass the full environment
#PBS -V
# send PBS output to /dev/null  (we redirect it below)
#PBS -o /dev/null
#PBS -e /dev/null
# ===== END PBS OPTIONS =====
#######################################################################

SHORT_JOBID=`echo $PBS_JOBID |cut -d. -f1`
exec 1>$PBS_O_WORKDIR/$PBS_JOBNAME-$SHORT_JOBID.out  2>$PBS_O_WORKDIR/$PBS_JOBNAME-$SHORT_JOBID.err
cd $PBS_O_WORKDIR

#Run code like on the command line. Ex:
#bash script.sh
#python script.py
#perl script.pl
#module load R
#R --no-save < script.R
#module load matlab
#matlab -nodisplay <script.m

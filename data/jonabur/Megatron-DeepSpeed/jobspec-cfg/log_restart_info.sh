#!/bin/bash
SCONTROL_OUT=$(scontrol show jobid $SLURM_JOBID)

regex="(Priority=[0-9]+).*(Restarts=[0-9]+).*(SubmitTime=.*)Deadline"
[[ $SCONTROL_OUT =~ $regex ]]
LOG_STRING="[$(date)] - $SLURM_JOB_NAME - $SLURM_JOBID - ${BASH_REMATCH[1]} - ${BASH_REMATCH[2]} - ${BASH_REMATCH[3]}"
echo $LOG_STRING
   


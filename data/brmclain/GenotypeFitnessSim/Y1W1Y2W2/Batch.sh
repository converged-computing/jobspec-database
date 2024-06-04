#!/bin/bash
#SBATCH  -J bmmclainMPCY1W1Y2W2       # job name
#SBATCH  -N 1                     # Number of nodes
#SBATCH  -n 48                    # number of task set to 1 for serial job, for parallel increase to 28, 48 for Sabine, or Carya respectively
#SBATCH  -o JobOutputDump/20231023/job_name.o%j          # output and error file name (%j expands to jobID)
#SBATCH  -t 25:00:00               # run time (hh:mm:ss)
#SBATCH  --mem-per-cpu=3750mb 

# Add the necessary programs to your working environment (optional, only use if you need 
module load python/3.9
cd /project/meisel/users/bmmclain/Y1W1Y2W2
python MultiProcessingCode_Y1W1Y2W2_v1.py $SLURM_JOB_ID 1000000 1000 48 random


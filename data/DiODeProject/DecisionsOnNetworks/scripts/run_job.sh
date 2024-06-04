#!/bin/bash

## Resource Request
#SBATCH --job-name=DDMnets           # name of your job, will be shown when running squeue
#SBATCH --output=log-out/DDMnets_%j.stdout   # name of the output file, %j will be replaced by the Job ID
#SBATCH --error=log-err/DDMnets_%j.stderr    # name of the error file, %j will be replaced by the Job ID
#SBATCH --partition=Epyc7452           # the hardware that you want to run on
#SBATCH --qos=short                    # the queue that you want to run on (short, long)
#SBATCH --ntasks=1                     # the job will launch a single task, set higher for MPI programs
#SBATCH --cpus-per-task=1              # each task will require 1 core on the same machine, set higher for OpenMP programs
#SBATCH --mail-user=andreagiovanni.reina@ulb.be   # your email to receive emails about the state of your job
#SBATCH --mail-type=END,FAIL           # when to send emails, choices are BEGIN, END, FAIL, ARRAY_TASKS

source /home/areina/pythonVirtualEnvs/DDMonNetsEnv/bin/activate
export PYTHONPATH=/home/areina/DecisionsOnNetworks/src/
cd $PYTHONPATH

srun python3 ${1} ${2}

deactivate

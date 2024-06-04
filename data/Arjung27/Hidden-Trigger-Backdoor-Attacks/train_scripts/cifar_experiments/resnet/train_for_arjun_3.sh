#!/bin/bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name=res3                            # sets the job name if not set from environment
#SBATCH --array=1                                 # Submit 8 array jobs, throttling to 4 at a time
#SBATCH --output cmllogs/%x_%A_%a.log                   # redirect STDOUT to; %j is the jobid, _%A_%a is array task id
#SBATCH --error cmllogs/%x_%A_%a.log                    # redirect STDERR to; %j is the jobid,_%A_%a is array task id
#SBATCH --account=scavenger                           # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger                                 # set QOS, this will determine what resources can be requested
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=scavenger
#SBATCH --mem 16gb                                      # memory required by job; MB will be assumed
#SBATCH --mail-user arjung15@umd.edu
#SBATCH --mail-type=END,TIME_LIMIT,FAIL,ARRAY_TASKS
#SBATCH --time=24:00:00                                 # how long will the job will take to complete; format=hh:mm:ss


CUDA_VISIBLE_DEVICES=0 python generate_poison.py cfg_CIFAR/singlesource_singletarget_binary_finetune/experiment_0013.cfg &&
CUDA_VISIBLE_DEVICES=0 python finetune_and_test.py cfg_CIFAR/singlesource_singletarget_binary_finetune/experiment_0013.cfg
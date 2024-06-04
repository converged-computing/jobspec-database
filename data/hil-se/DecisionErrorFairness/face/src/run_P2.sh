#!/bin/bash -l
## NOTE the -l flag!

## Name of the job -You'll probably want to customize this
#SBATCH -J scut_face

## Use the resources available on this account
#SBATCH -A loop

## To send mail for updates on the job
#SBATCH --mail-user=zxyvse@rit.edu
#SBATCH --mail-type=ALL

## Standard out and Standard Error output files
#SBATCH -o log/%J_%a.o
#SBATCH -e log/%J_%a.e

## Request 1 Days, 0 Hours, 0 Minutes, 0 Seconds run time MAX,
## anything over will be KILLED
#SBATCH -t 1-00:00:00

## Put in tier3 partition for testing small jobs, like this one
## But because our requested time is over 4 day, it won't run, so
## use any tier you have available
#SBATCH -p tier3

## Request 1 GPU for one task, note how you can put multiple commands
## on one line
#SBATCH --gres=gpu:a100:4

## Job memory requirements in MB
#SBATCH --mem=40G

## Job script goes below this line

spack unload -a
## Load modules with spack
## Tensorflow
spack load /xi3pch3
## keras
spack load py-keras
## Execute target code
python3 fb_main.py run P2




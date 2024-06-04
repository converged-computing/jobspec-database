#!/bin/bash
#
#SBATCH --partition=c2_cpu
#SBATCH --ntasks=1
#SBATCH --mem=12000
#SBATCH --nodes=1
#SBATCH --workdir=/media/labs/rsmith/lab-members/cgoldman/Wellbeing/advise_task/scripts
#SBATCH --begin=now
#SBATCH --job-name=advice-fit
#
#################################################

SUBJECT=$1
export SUBJECT

INPUT_DIRECTORY=$2
export INPUT_DIRECTORY

RESULTS=$3
export RESULTS

module load matlab/2022a
run_file='/media/labs/rsmith/lab-members/cgoldman/Wellbeing/advise_task/scripts/main_advise_recover.m'
matlab -nodisplay -nosplash < ${run_file}
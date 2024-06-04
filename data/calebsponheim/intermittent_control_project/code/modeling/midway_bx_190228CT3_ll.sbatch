#!/bin/bash
#SBATCH --array=2-30
#SBATCH --job-name=Bx_ll_sbatch_%A
#SBATCH --output=./data_midway/log_files/Bx190228CT3_%a.out
#SBATCH --error=./data_midway/log_files/Bx190228CT3_%a.err
#SBATCH --partition=broadwl
#SBATCH --ntasks=4
#SBATCH --mem=16G

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

module load matlab/2014b
matlab -nojvm -nodisplay -nosplash -r "addpath(genpath('./')); midway_train_HMM_log_likelihood('./data/Bx190228CT3.mat',$SLURM_ARRAY_TASK_ID,4)"
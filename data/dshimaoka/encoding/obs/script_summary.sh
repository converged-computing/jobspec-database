#!/bin/bash
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=daisuke.shimaoka@monash.edu
#SBATCH --job-name=Summary
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=12000
#SBATCH --gres=gpu:1
#SBATCH --partition=m3g
module load matlab
matlab -nodisplay -nodesktop -nosplash < summaryAcrossPix.m

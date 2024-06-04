#!/bin/bash
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=elizabeth.zavitz@monash.edu
#SBATCH --job-name=procSpike
#SBATCH --time=00:15:00
#SBATCH --ntasks=6
#SBATCH --mem=30000
#SBATCH --array=1-384

module load matlab/r2021a
matlab -nodisplay -nojvm -nosplash < /home/earsenau/code/processSpikingMoveStim/RUNAnalysis_mdbExtract.m

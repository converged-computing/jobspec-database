#!/bin/bash -l
#SBATCH --job-name=rbatch_1
#SBATCH --array=1-73
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
#SBATCH --partition=cpu
#SBATCH --mem-per-cpu=40000MB
#SBATCH --mail-type=END,FAIL
# script to submit an r batch job

number=$SLURM_ARRAY_TASK_ID
module load r/4.1.1-gcc-9.4.0-withx-rmath-standalone-python-3.8.12
Rscript 08_shearwaterML.R $number


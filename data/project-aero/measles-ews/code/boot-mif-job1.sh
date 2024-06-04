#!/bin/bash
#SBATCH --array=1-1000
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --constraint="intel"
#SBATCH --ntasks-per-node 1
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=./reports/slurm-%A_%a.out
 
cd ~/measles/code/
module load R
 
# parallel R: submit job with one MPI master
R CMD BATCH -$SLURM_ARRAY_TASK_ID bootstrap-fit-mif.R

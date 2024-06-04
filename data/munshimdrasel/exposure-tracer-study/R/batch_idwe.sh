#!/bin/sh

#SBATCH --job-name=batch_idwe
#SBATCH --partition=bigmem
#SBATCH --constraint=amd
#SBATCH --output=/scratch/%u/slurm_scipts/idwe_%j.out  # Output file
#SBATCH --error=/scratch/%u/slurm_scipts/idwe_%j.err   # Error file

#SBATCH --mail-type=BEGIN,END,FAIL         # ALL,NONE,BEGIN,END,FAIL,REQUEUE,..
#SBATCH --mail-user=mrasel@gmu.edu     # Put your GMU email address here
#SBATCH --mem-per-cpu=16G        # Total memory needed per task (units: K,M,G,T)

## Specify how much time your job needs. (default: see partition above)
#SBATCH --time=07-00:00  # Total time needed for job: Days-Hours:Minutes

## create an array of jobs
#SBATCH -a 1-10

## select a specific node
## SBATCH --nodelist=hop[071, 073]

## Load the relevant modules needed for the job

module load gnu10/10.3.0
module load openmpi
module load netcdf-c
module load r/4.1.2-dx


#going to R script directory
cd /projects/HAQ_LAB/mrasel/R/exposure-tracer-study/R/

## Run your program or script
Rscript ./inverse_distance.R   


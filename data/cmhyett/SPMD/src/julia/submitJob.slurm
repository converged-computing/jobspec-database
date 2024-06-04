#!/bin/bash
 
# --------------------------------------------------------------
### PART 1: Requests resources to run your job.
# --------------------------------------------------------------
#SBATCH --job-name=SPMD
### SLURM reads %x as the job name and %j as the job ID
#SBATCH --output=%x.out
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=standard
#SBATCH --ntasks=6
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4gb
#SBATCH --time=1:00:00
#SBATCH --array=1-10
 
# --------------------------------------------------------------
### PART 2: Executes bash commands to run your job
# --------------------------------------------------------------
### Load required modules/libraries if needed
module load julia
PROJECT_PATH=${HOME}/SPMD/src/julia/
OUTPUT_PATH=${PROJECT_PATH}/results/${SLURM_ARRAY_TASK_ID}/
MAX_EPOCHS=500
julia --project=${PROJECT_PATH} ${PROJECT_PATH}/runScript.jl ${OUTPUT_PATH} ${MAX_EPOCHS}

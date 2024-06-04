#!/bin/bash
 
#SBATCH -N 1  # number of nodes
#SBATCH -c 8  # number of cores to allocate
#SBATCH --mem=64G
#SBATCH -t 0-09:00:00   # time in d-hh:mm:ss
#SBATCH -p general       # partition 
#SBATCH --gres=gpu:1
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment

# Always purge modules to ensure consistent environments
module purge    
# Load required modules for job's environment
module load mamba 

source activate my_vima 

cd /home/rlcorrea/CSE574_project_vima/VIMA/scripts
python3 behavior_cloning_batched_v3.py

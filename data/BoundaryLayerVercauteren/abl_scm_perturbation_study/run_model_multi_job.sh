#!/bin/bash

#SBATCH --account=account_name

# Runtime and memory
#SBATCH --mem=10GB
#SBATCH --time=05:00:00
#SBATCH --array=1-50

#SBATCH --ntasks-per-node=128 # number of cores, max 128 on fox

singularity exec -H /fp/projects01/account_name/abl_scm_perturbation_study/ docker/abl_scm_venv.sif python3 -u main.py $SLURM_ARRAY_TASK_ID 50

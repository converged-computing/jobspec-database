#!/bin/bash

#SBATCH -n 1
#SBATCH -t 36:00:00
#SBATCH --mem 4096MB 
#SBATCH --tmp=4096MB
#SBATCH --job-name ANNz_Regression
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.luken@westernsydney.edu.au
#SBATCH --output slurm_logs/annz_%A_%a.out
#SBATCH --error slurm_logs/annz_%A_%a.err
#SBATCH --array=0-99

source /fred/oz237/kluken/redshift_pipeline_adacs/Slurm/hpc_profile_setup.sh

mkdir seed_${SLURM_ARRAY_TASK_ID}
cd seed_${SLURM_ARRAY_TASK_ID}

singularity run $container_path/annz_latest.sif $script_path/annz.py -s ${SLURM_ARRAY_TASK_ID}  -l

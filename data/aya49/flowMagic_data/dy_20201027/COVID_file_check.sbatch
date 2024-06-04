#!/bin/bash 
#SBATCH -o COVID_FILE_CHECK_%A_%a.out

singularity exec -B /mnt/f:/data/ \
/mnt/f/Docker/BrinkmanLabSingularity/brinkman_lab_singularity_190418.im Rscript \
/data/Brinkman\ group/COVID/data/code/data_check.R ${SLURM_ARRAY_TASK_ID}

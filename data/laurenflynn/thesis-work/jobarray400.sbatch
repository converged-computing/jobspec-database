#!/bin/bash
#SBATCH -J generateFamilies # A single job name for the array
#SBATCH -p serial_requeue # Partition
#SBATCH -c 1 # one core
#SBATCH -t 0-12:30 # Running time of 2.5 hours
#SBATCH --mem 20000 # Memory request of 20 gb
#SBATCH -o slurm_log_files/family_generation/10000families_%A_%a.out # Standard output
#SBATCH -e slurm_log_files/family_generation/10000families_%A_%a.err # Standard error
#SBATCH --array=1-400
singularity exec --cleanenv --env R_LIBS_USER=$HOME/R/ifxrstudio/RELEASE_3_15 /n/singularity_images/informatics/ifxrstudio/ifxrstudio:RELEASE_3_15.sif Rscript src/2_slurm_with_loops.R ${SLURM_ARRAY_TASK_ID}

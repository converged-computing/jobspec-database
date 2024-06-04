#!/bin/bash
## run the Rscript 02_Setbp1_vision_kidney_hallmarks.R and schedule this job to SLURM


#SBATCH --job-name=kid_vis_hall
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jwhitlock@uab.edu #CHANGE THE EMAIL TO YOUR OWN
#
#Number of tasks needed for this job, the partition, and parameters
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=255000
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --partition=short #partition info here: https://docs.rc.uab.edu/cheaha/hardware/#partitions
#
#SBATCH --output=VISION_kid_hall.out
#SBATCH --error=VISION_kid_hall.err

########################################
### PUT YOUR COMMANDS BELOW THIS BOX ###
########################################

#load modules
module load Singularity/3.5.2-GCC-5.4.0-2.26

#variables
wd="/data/user/jbarham3/230227_JW_Setbp1Manuscript"
src="/data/user/jbarham3/230227_JW_Setbp1Manuscript/src/module_scores" #be sure that your subdirectories are structured the same

export SINGULARITYENV_PASSWORD='pass'
export SINGULARITYENV_USER='jbarham3'

#code to execute docker and script for analysis
cd ${wd}

singularity exec --cleanenv --no-home -B ${wd} ${wd}/bin/docker/setbp1_manuscript_1.0.11.sif Rscript --vanilla ${src}/02_Setbp1_vision_kidney_hallmarks.R # here vanilla ensures only the script is run and environment is kept clean
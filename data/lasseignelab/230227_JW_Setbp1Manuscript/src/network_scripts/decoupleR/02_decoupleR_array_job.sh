#!/bin/bash
## run the Rscript decoupleR_analysis.R and schedule this job to SLURM with
## `sbatch decoupleR_array_job.sh`

#SBATCH --job-name=kid_decoupleR #Note: when running kidney this needs to be changed to 'kid_decoupleR' from 'cor_decoupleR'
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jbarham3@uab.edu #CHANGE THE EMAIL TO YOUR OWN
#
#Number of tasks needed for this job, the partition, and parameters
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=255000
#SBATCH --nodes=1
#SBATCH --time=50:00:00 #Note: '50:00:00 hours for brain, when running kidney this needs to be changed to 5 days as '120:00:00' (this may end up being 120 for both because of ex neurons)
#SBATCH --partition=largemem #Note: when running brain, can use largemem. Partition info here: https://docs.rc.uab.edu/cheaha/hardware/#partitions
#
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --array=0-34 #Note: when running kidney this needs to be changed to 0-34 from 0-16 for brain; 16 dfs in list for cortex and 34 for kidney

########################################
### PUT YOUR COMMANDS BELOW THIS BOX ###
########################################
module load R
module load Singularity/3.5.2-GCC-5.4.0-2.26

wd=/data/user/jbarham3/230227_JW_Setbp1Manuscript #change this to match your project directory path
#exp="/data/project/lasseigne_lab/JordanWhitlock/230227_Setbp1Manuscript/processed_data/decoupleR_expression_inputs" #be sure that your subdirectories are structured the same

export SINGULARITYENV_PASSWORD='pass'
export SINGULARITYENV_USER='jbarham3' #change this to your user


cd ${wd}

#input expression .Rdata file
#RDATA_FILE=${wd}/data/processed/decoupleR_expression_inputs/cerebral_expression.Rdata # NOTE: be sure to uncomment 'cerebral_expression' for 'kidney_expression', or other tissue of interest and resubmit this same job
RDATA_FILE=${wd}/data/processed/decoupleR_expression_inputs/kidney_expression.Rdata
echo "Opening ${RDATA_FILE}"

#input prior network file
PRIOR_NET=${wd}/data/processed/decoupleR_prior_CollecTRI/mouse_prior_tri.csv
echo "Prior Network from ${PRIOR_NET}"

ITEM=$(Rscript -e "load('${RDATA_FILE}'); cat(names(decoupleR_expression)[${SLURM_ARRAY_TASK_ID}])")
echo "Processing: ${ITEM}"

#TISSUE="brain" #uncomment this to "kidney" from "brain" where appropriate
TISSUE="kidney"
MIN_N=5

singularity exec --cleanenv --no-home -B ${wd} ${wd}/bin/docker/setbp1_manuscript_1.0.6.sif Rscript --vanilla ${wd}/src/network_scripts/decoupleR/02_decoupleR_analysis.R ${RDATA_FILE} ${PRIOR_NET} ${ITEM} ${TISSUE} ${MIN_N}

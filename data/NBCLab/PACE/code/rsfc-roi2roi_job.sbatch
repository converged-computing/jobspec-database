#!/bin/bash
#SBATCH --job-name=rsfc
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4gb
#SBATCH --partition=short
# Outputs ----------------------------------
#SBATCH --output=log/%x/COC100/%x-COC100_%A-%a.out   
#SBATCH --error=log/%x/COC100/%x-COC100_%A-%a.err   
# ------------------------------------------


pwd; hostname; date
set -e

# sbatch --array=2-$(( $( wc -l ../dset-COC100/participants.tsv | cut -f1 -d' ' ) - 1 ))%2 rsfc_job.sbatch

#==============Shell script==============#
#Load the software needed
module load singularity/3.7.1
#spack load singularity@3.7.1

DATA="COC100"
HOST_DIR="/gpfs1/home/m/r/mriedel"
PROJECT="pace"
DSETS_DIR="${HOST_DIR}/${PROJECT}/dsets"
BIDS_DIR="${DSETS_DIR}/dset-${DATA}"
IMG_DIR="${HOST_DIR}/${PROJECT}/software"
DERIVS_DIR="${BIDS_DIR}/derivatives"
ATLAS_DIR="${HOST_DIR}/${PROJECT}/atlas"

mriqc_ver=0.16.1
afni_ver=16.2.07
fmriprep_ver=20.2.5
MRIQC_DIR="${DERIVS_DIR}/mriqc-${mriqc_ver}"
CLEAN_DIR="${DERIVS_DIR}/denoising-${afni_ver}"

subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" ${BIDS_DIR}/participants.tsv )

desc_clean="aCompCorCens"
desc_sm="aCompCorSM6Cens"
space="MNI152NLin2009cAsym"

# Run RSFC pipeline
RSFC_DIR="${DERIVS_DIR}/rsfc-roi2roi"
mkdir -p ${RSFC_DIR}

# Run denoising pipeline
# Compose the command line
SHELL_CMD="singularity exec --cleanenv \
    -B ${DSETS_DIR}/code:/code \
    -B ${CLEAN_DIR}:/clean \
    -B ${RSFC_DIR}:/rsfc \
    -B ${ATLAS_DIR}:/atlas_dir \
    $IMG_DIR/afni-22.0.20.sif"
# Run python script inside fmriprep environment
rsfc="${SHELL_CMD} python /code/rsfc-roi2roi.py \
    --clean_dir /clean \
    --rsfc_dir /rsfc \
    --atlas_dir /atlas_dir \
    --subject sub-${subject} \
    --space ${space} \
    --desc_list ${desc_clean} ${desc_sm} \
    --n_jobs ${SLURM_CPUS_PER_TASK}"
# Setup done, run the command

echo
echo Commandline: $rsfc
eval $rsfc 
exitcode=$?

# Output results to a table
echo "sub-$subject   ${SLURM_ARRAY_TASK_ID}    $exitcode" \
      >> ${DSETS_DIR}/code/log/${SLURM_JOB_NAME}/${DATA}/${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv
echo Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code $exitcode
date

exit $exitcode
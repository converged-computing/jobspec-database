#!/bin/bash
#SBATCH --job-name=lpp
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4gb
#SBATCH --account=iacc_nbc
#SBATCH --qos=pq_nbc
#SBATCH --partition=IB_40C_512G
# Outputs ----------------------------------
#SBATCH --output=/home/data/nbc/external-datasets/ds003643/code/jobs/fmriprep-v21-%a.out
#SBATCH --error=/home/data/nbc/external-datasets/ds003643/code/jobs/fmriprep-v21-%a.err
# ------------------------------------------
# Submit this job with the array option, starting with 1.
# sbatch --array=1-119%5 fmriprep_job_v21.sbatch
pwd; hostname; date
set -e

#==============Shell script==============#
#Load the software needed
module load singularity-3.5.3

IMG_DIR="/home/data/cis/singularity-images"
DATA_DIR="/home/data/nbc"
BIDS_DIR="${DATA_DIR}/external-datasets/ds003643"
CODE_DIR="${DATA_DIR}/external-datasets/ds003643/code"

DERIVS_DIR="${BIDS_DIR}/derivatives/fmriprep-v21.0.0"
mkdir -p ${DERIVS_DIR}

# Parse the participants.tsv file and extract one subject ID from the line corresponding to this SLURM task.
subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" ${BIDS_DIR}/participants.tsv )

WORK_DIR="${BIDS_DIR}/work/fmriprep-${subject}"
mkdir -p ${WORK_DIR}

# Prepare some writeable bind-mount points.
TEMPLATEFLOW_HOST_HOME="${HOME}/.cache/templateflow"
FMRIPREP_HOST_CACHE="$HOME/.cache/fmriprep"
mkdir -p ${FMRIPREP_HOST_CACHE}

# Make sure FS_LICENSE is defined in the container.
FS_LICENSE="${HOME}/freesurfer_license.txt"

# Designate a templateflow bind-mount point
export SINGULARITYENV_TEMPLATEFLOW_HOME="$HOME/.cache/templateflow"

SINGULARITY_CMD="singularity run --cleanenv \
      -B ${BIDS_DIR}:/data \
      -B ${DERIVS_DIR}:/out \
      -B ${TEMPLATEFLOW_HOST_HOME}:${SINGULARITYENV_TEMPLATEFLOW_HOME} \
      -B ${WORK_DIR}:/work \
      ${IMG_DIR}/poldracklab-fmriprep_21.0.0.sif"

# Compose the command line
cmd="${SINGULARITY_CMD} /data \
      /out \
      participant \
      --participant-label $subject \
      -w /work/ \
      -vv \
      --omp-nthreads 8 \
      --nprocs 8 \
      --mem_mb 32000 \
      --output-spaces MNI152NLin6Asym:res-native anat:res-native func:res-native \
      --me-output-echos \
      --notrack \
      --stop-on-first-crash \
      --skip_bids_validation \
      --fs-license-file $FS_LICENSE"

# Setup done, run the command
echo Running task ${SLURM_ARRAY_TASK_ID}
echo Commandline: $cmd
eval $cmd
exitcode=$?

# Clean up the working directory
rm -rf ${WORK_DIR}

# Output results to a table
echo "sub-$subject   ${SLURM_ARRAY_TASK_ID}    $exitcode" \
      >> ${CODE_DIR}/jobs/${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv
echo Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code $exitcode
exit $exitcode

date


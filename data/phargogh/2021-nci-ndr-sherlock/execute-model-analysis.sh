#!/bin/bash
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --partition=hns,normal
#SBATCH --mail-type=ALL
#SBATCH --job-name="NCI-NOXN-model-analysis"
#SBATCH --output=/scratch/users/jadoug06/slurm-logfiles/slurm-%j.%x.out

set -ex

# "${ENVVAR:-DEFAULT}" means use ENVVAR if present, DEFAULT if not.
MODEL_ANALYSIS_WORKSPACE="${NOXN_WORKSPACE:-$1}"  # final location of pipeline outputs
NCI_WORKSPACE="${NCI_WORKSPACE:-$2}"
RESOLUTION="$3"
if [ "$RESOLUTION" = "" ]
then
    echo "Parameter 3 must be the resolution"
    exit 1
fi

# load configuration for globus, singularity, etc.
echo "$(pwd)"
source "env-sherlock.env"

pushd nci-noxn-levels

DATE="$(date +%F)"
GIT_REV="rev$(git rev-parse --short HEAD)"

CONFIG_FILE="pipeline.config-sherlock-$RESOLUTION.json"
singularity run \
    "docker://$NOXN_DOCKER_CONTAINER" \
    python3 model_analysis.py \
    --n_workers "$SLURM_CPUS_PER_TASK" \
    "$CONFIG_FILE" \
    "$MODEL_ANALYSIS_WORKSPACE" \
    "$NCI_WORKSPACE/ndr-plus-outputs/"

# Upload files to globus
# Copy geotiffs AND logfiles, if any, to google drive.
# This will upload to a workspace with the same dirname as $MODEL_ANALYSIS_WORKSPACE.
module load system py-globus-cli
ARCHIVE_DIR="$DATE-nci-noxn-model-analysis-$GIT_REV-slurm$SLURM_JOB_ID"
GDRIVE_DIR="nci-ndr-stanford-gdrive:$(basename $NCI_WORKSPACE)/$ARCHIVE_DIR"
globus transfer --fail-on-quota-errors --recursive \
    --label="NCI WQ NOXN Model Analysis $GIT_REV" \
    "$GLOBUS_SHERLOCK_SCRATCH_ENDPOINT_ID:$MODEL_ANALYSIS_WORKSPACE" \
    "$GLOBUS_STANFORD_GDRIVE_COLLECTION_ID:$GLOBUS_STANFORD_GDRIVE_RUN_ARCHIVE/$(basename $NCI_WORKSPACE)/$ARCHIVE_DIR" || echo "Globus transfer failed!"

echo "Model analysis complete."

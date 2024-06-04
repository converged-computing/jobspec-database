#!/bin/bash
#SBATCH --gpus 1
#SBATCH --mem 32G
#SBATCH --cpus-per-gpu 8
#SBATCH --time 16:0:0
#SBATCH --partition short


CONTAINER="${HOME}/containers/rapids-prod.sif"
CONTAINER_RC_FILE="${HOME}/containers/singularity_rc"
SCRIPT="./run_dask.sh"

singularity run --nv -B /scratch/shared/pwesolowski,/run/udev:/run/udev:ro "$CONTAINER" /bin/bash --rcfile "$CONTAINER_RC_FILE" -ci "$SCRIPT"

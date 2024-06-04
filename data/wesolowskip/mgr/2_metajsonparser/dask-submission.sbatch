#!/bin/bash
#SBATCH --gpus 1
#SBATCH --mem 32G
#SBATCH --cpus-per-gpu 8
#SBATCH --time 5-0:0:0
#SBATCH --partition long


CONTAINER="${HOME}/containers/rapids-prod.sif"
CONTAINER_RC_FILE="${HOME}/containers/singularity_rc"
SCRIPT="python -u dask_benchmark.py"

singularity run --nv -B /scratch/shared/pwesolowski,/run/udev:/run/udev:ro "$CONTAINER" /bin/bash --rcfile "$CONTAINER_RC_FILE" -ci "$SCRIPT"

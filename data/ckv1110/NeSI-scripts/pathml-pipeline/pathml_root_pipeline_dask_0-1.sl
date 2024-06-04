#!/bin/bash -e
#SBATCH --job-name=pathml_dask_tile-only
#SBATCH --time=02:00:00
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12

module unload XALT
module load Singularity

# Bind directories and append SLURM job ID to output directory
export SINGULARITY_BIND="/nesi/project/uoa03709/work-dir:/var/inputdata"
export NAME=TCGA-02-0003-01Z-00-DX1.6171b175-0972-4e84-9997-2f1ce75f4407Region01

# Run container %runscript
srun singularity exec /nesi/project/uoa03709/containers/sif/smp-cv_0.1.6.sif env PYTHONUNBUFFERED=1 python /var/inputdata/py-data/root_pipeline_dask.py
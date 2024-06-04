#!/bin/bash -e
#SBATCH --job-name=pathml_root_pipeline-dask
#SBATCH --time=12:00:00
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12

module unload XALT
module load Singularity

# Bind directories and append SLURM job ID to output directory
mkdir /nesi/nobackup/uoa03709/output/${SLURM_JOB_ID:-0}
export SINGULARITY_BIND="/nesi/nobackup/uoa03709/input:/var/inputdata/work-dir,\
/nesi/nobackup/uoa03709/output/${SLURM_JOB_ID:-0}:/var/outputdata"

# Run container %runscript
srun singularity exec /nesi/project/uoa03709/containers/sif/smp-cv_0.1.6.sif env PYTHONUNBUFFERED=1 python /var/inputdata/root_pipeline_dask.py

#!/bin/bash -e
#SBATCH --job-name=pathml_root_pipeline-TCGA-14-0789-01Z-00-DX6
#SBATCH --time=12:00:00
#SBATCH --hint=multithread
#SBATCH --partition=bigmem
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12

module load Singularity

# Bind directories and append SLURM job ID to output directory
mkdir /nesi/nobackup/uoa03709/output/${SLURM_JOB_ID:-0}
export SINGULARITY_BIND="/nesi/nobackup/uoa03709/input:/var/inputdata,\
/nesi/nobackup/uoa03709/output/${SLURM_JOB_ID:-0}:/var/outputdata"

# Run container %runscript
srun singularity exec smp-cv_0.1.3.sif python /var/inputdata/root_pipeline_dask.py
#!/bin/bash -l

#SBATCH --job-name=Nextflow-master-BLAST
#SBATCH --account=pawsey0001
#SBATCH --partition=workq
#SBATCH --ntasks=1
#SBATCH --time=00:30:00

module load singularity  # just in case image pull is needed
module load nextflow

nextflow run blast.nf -profile zeus

ls -ltr

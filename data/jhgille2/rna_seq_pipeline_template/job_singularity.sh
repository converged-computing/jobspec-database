#!/bin/bash

# Submit the pipeline as a job with srun job.sh

# Modified from https://github.com/mschubert/clustermq/blob/master/inst/LSF.tmpl
# under the Apache 2.0 license:
#SBATCH --job-name=soy_test_alignment
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --mem-per-cpu=3000
#SBATCH --cpus-per-task=1
#SBATCH -N 1
#SBATCH -n 72
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH --mail-user=jhgille2@ncsu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

module load singularity
singularity exec conda.sif R CMD BATCH run.R

# Removing .RData is recommended.
# rm -f .RData

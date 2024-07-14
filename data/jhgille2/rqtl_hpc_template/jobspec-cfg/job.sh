#!/bin/bash

# Submit the pipeline as a job with srun job.sh

# Modified from https://github.com/mschubert/clustermq/blob/master/inst/LSF.tmpl
# under the Apache 2.0 license:
#SBATCH --job-name=raleigh_soja_qtl_mapping
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --mem-per-cpu=3000
#SBATCH --cpus-per-task=1
#SBATCH -N 4
#SBATCH -n 287
#SBATCH -t 40:00:00
#SBATCH --mail-user=jhgille2@ncsu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --exclusive
#SBATCH -p=short

module load R # Comment out if R is not an environment module.
module load miniconda
source activate r_qtl_hpc
R CMD BATCH run.R

# Removing .RData is recommended.
# rm -f .RData

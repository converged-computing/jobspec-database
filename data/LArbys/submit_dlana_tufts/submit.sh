#!/bin/bash

# slurm submission script for making larmatch training data

#SBATCH --job-name=dlana
#SBATCH --output=dlana.log
#SBATCH --mem-per-cpu=2000
#SBATCH --time=30:00
#SBATCH --array=0-1657

container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_dldependencies_pytorch1.3.sing
RUN_DLANA_DIR=/cluster/tufts/wongjiradlab/larbys/run_dlana_jobs


module load singularity
srun singularity exec ${container} bash -c "cd ${RUN_DLANA_DIR} && source run.sh"


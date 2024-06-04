#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./fitlogs/job.out.%j
#SBATCH -e ./fitlogs/job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J test_slurm
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#
# Wall clock limit (max. is 24 hours):
#SBATCH -p long
#SBATCH --time=48:00:00
#SBATCH --mem=8000MB

module purge
module load apptainer
export SINGULARITY_TMPDIR=$(pwd)/tmp
export SINGULARITY_CACHEDIR=$(pwd)/tmp

set -x
mkdir -p CABCHSV fitresults pseudodata

srun  singularity exec -B $(pwd):$(pwd) --env JULIA_DEPOT_PATH=$(pwd)/J:/opt/julia docker://ghcr.io/mppmu/high-x-parton-pdf-inference:latest $JULIA  "$@"
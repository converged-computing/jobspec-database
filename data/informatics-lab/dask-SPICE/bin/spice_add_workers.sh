#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH --mem=200000
#SBATCH --ntasks=48
#SBATCH --error=/scratch/dkillick/SPICE/dask/job-%N-%j.err
#SBATCH --output=/scratch/dkillick/SPICE/dask/job-%N-%j.log
#SBATCH --time=60


# Start new dask workers on a SPICE node and add them to an existing scheduler.

module load scitools

HOST=${1}
PORT=${2}
NWORKERS=48

dask-worker --nprocs ${NWORKERS} --nthreads 1 "${HOST}:${PORT}"
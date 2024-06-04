#!/bin/bash

#SBATCH --job-name=dask
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:01:00


module load Singularity
module unload XALT

srun singularity run -B $PWD $SIFPATH/dask-mpi_latest.sif dask_example.py

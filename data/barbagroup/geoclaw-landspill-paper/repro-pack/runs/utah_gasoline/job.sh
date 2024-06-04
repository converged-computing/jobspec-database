#!/bin/bash

#SBATCH --job-name="Landspill Utah Gasoline"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-job-%j.out
#SBATCH --error=slurm-job-%j.err
#SBATCH --partition=defq


# add custom Lmod module file path
module use /lustre/groups/barbalab/modulefiles

# load mpi, singularity, and the collections
module load singularity/3.4.2
module load singularity-collections/1.0.0
module list

# image
IMAGE=${SINGULARITY_COLLECTIONS}/landspill-bionic.sif
ROOT_DIR=/lustre/groups/barbalab/pychuang/proposal-simulations/landspill-runs
CASE_NAME=utah_gasoline
CASE_DIR=${ROOT_DIR}/${CASE_NAME}

# disable using file locking on lustre for HDF5
export HDF5_USE_FILE_LOCKING=FALSE

# experience shows using 20 cores is faster
export OMP_NUM_THREADS=20

# check the environment variables
env > ${CASE_DIR}/env_vars.log

# run case
cd $CASE_DIR
echo "Currently running $CASE_NAME"

singularity run --app run ${IMAGE} ${CASE_DIR} > ${CASE_DIR}/stdout-${SLURM_JOBID}.log 2>&1 &
wait

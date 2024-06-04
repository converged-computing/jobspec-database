#!/bin/bash
#SBATCH --account ffnr0871-rfi-test
#SBATCH --qos rfi
#SBATCH --time 60
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 36
#SBATCH --gpus-per-task 0
#SBATCH --job-name logs/build-mpi4jax
#SBATCH -o %x-%j-%t-%N.out

module purge
module load baskerville
module load bask-apps/live
module load OpenMPI/4.0.5-gcccuda-2020b

set -x

export ROOTDIR="/bask/projects/f/ffnr0871-rfi-test/pje39613"
export SINGULARITY_CACHEDIR="$ROOTDIR/.singularity-cache"

export JOBDIR="$ROOTDIR/container-jax/examples/baskerville/build-mpi4jax"
export WHEELSDIR="$JOBDIR/wheels"

export MPIDIR="`dirname $(which mpicc)`/../"
export SINGULARITYENV_PREPEND_PATH="/usr/local/mpi/bin:/usr/local/mpi"
export SINGULARITYENV_PREPEND_LD_LIBRARY_PATH="/usr/local/mpi/lib64:/usr/local/mpi/lib:/usr/local/mpi"

mpirun singularity run --nv \
                       -B $WHEELSDIR:/wheels \
                       -B $MPIDIR:/usr/local/mpi \
                       --env MPICC=/usr/local/mpi/bin/mpicc \
                       --env CUDA_ROOT=/usr/local/cuda \
                       docker://quay.io/rosalindfranklininstitute/jax:v0.3.1-devel \
                   pip wheel --no-cache-dir -w /wheels mpi4py mpi4jax

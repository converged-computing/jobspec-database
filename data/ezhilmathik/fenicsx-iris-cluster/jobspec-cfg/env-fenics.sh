#!/bin/bash
DIR=$(dirname $(readlink -f "${BASH_SOURCE}"))
source ${DIR}/env-build-fenics.sh

# Note: These must be unset initially, need to code this in somehow.
export PETSC_DIR=${PREFIX}

source ${PREFIX}/python-venv/bin/activate

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

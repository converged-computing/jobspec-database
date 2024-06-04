#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --job-name=gwecc-search

# module load pkgsrc/2022Q2

/home/susobhan/Data/susobhan/miniconda/envs/gwecc/bin/activate

PYTHON=$CONDA_PREFIX/bin/python
JULIA=$CONDA_PREFIX/bin/julia

source print_info.sh
# mpichversion | head -n 1
# $PYTHON -c 'import mpi4py as mpi; print("mpi4py version", mpi.__version__)'

echo

$PYTHON run_1psr_analysis.py fix-wn_vary-rn_no-ecw.json

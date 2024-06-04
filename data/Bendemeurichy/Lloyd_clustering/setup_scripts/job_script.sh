#!/bin/bash

#PBS -N lloyd_benches                   ## job name
#PBS -l nodes=1:ppn=4               ## 1 nodes, 4 cores per node
#PBS -l walltime=6:00:00            ## max. 6h of wall time
#PBS -l mem=32gb                    ## 32GB of memory
#PBS -m abe                         ## send mail on abort, begin and end

PIP_DIR="$VSC_SCRATCH/site-packages" # directory to install packages
CACHE_DIR="$VSC_SCRATCH/.cache" # directory to use as cache

# Load PyTorch
module load Python/3.10.4-GCCcore-11.3.0

# activate venv
source venv_doduo/bin/activate







# Start script
cd $PBS_O_WORKDIR

rm -rf "$VSC_DATA/data"
rm -rf "$VSC_DATA/output"

mkdir "$VSC_DATA/data"
mkdir "$VSC_DATA/output"

#PYTHONPATH="$PYTHONPATH:$PIP_DIR" python exp1.py --quality 20 --only_adv


PYTHONPATH="$PYTHONPATH:$PIP_DIR" python ~/compbio_lloyd/notebooks/test_and_bench.py
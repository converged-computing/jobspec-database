#!/bin/bash

#-------------------------------------------------------
# SBATCH -J build_cmdstan  # sed_* will be replaced w/ sed on creation of experiment
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p vm-small
#SBATCH -o logs/build_stan.o
#SBATCH -e logs/build_stan.e
#SBATCH -t 00:20:00 

#------------------------------------------------------


cmdstan_ver=2.31.1
make_cores=6

IMAGE=$WORK/singularity/edge-trait-meta_4.2.0.sif # singularity image


cmdstan_direc=$SCRATCH/cmdstan # Feel free to change to $WORK, though that will require modifying the scripts that call it.
cmdstan_temp=$cmdstan_direc # If you change cmdstan_direc to $WORK, change this to download temp files elsewhere.
tarball="cmdstan-${cmdstan_ver}.tar.gz"

# Download and extract the cmdstan file
url="https://github.com/stan-dev/cmdstan/releases/download/v${cmdstan_ver}/${tarball}"
cd $cmdstan_tmp
wget $url 
tar -xf $tarball -C $cmdstan_direc

# Setup and prep for installation
cd $cmdstan_direc/cmdstan-${cmdstan_ver}
# Set up the make/local flags for compiler

echo CXXFLAGS += -march=native -mtune=native -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE >> make/local
echo LDLIBS += -lblas -llapack  >> make/local # -llapacke
echo STAN_THREADS = true >> make/local
# NOTE: The following lines of code use the Intel MKL, which is non-free software
# If you include them, please be sure to register for a free community license
echo CXXFLAGS += -DEIGEN_USE_MKL_ALL -I"/usr/include/mkl" >> make/local
echo LDLIBS += -lmkl_intel_lp64 -lmkl_sequential -lmkl_core >> make/local

# Compile cmdstan using the docker image

export LD_PRELOAD=""
module load tacc-apptainer
module unload xalt
HDIR=/home
singularity exec -H $HDIR $IMAGE make -j${make_cores} build

# If you wish to check the installation
# singularity exec -H $HDIR $IMAGE make examples/bernoulli/bernoulli
# singularity exec -H $HDIR $IMAGE ./examples/bernoulli/bernoulli sample data file=examples/bernoulli/bernoulli.data.json
# ls -l output.csv

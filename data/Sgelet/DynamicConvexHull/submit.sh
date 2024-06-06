#!/bin/sh
### General options
### -- specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J DynConHull
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need 2GB of memory per core/slot --
#BSUB -R "rusage[mem=64GB]"
### -- Request specific CPU
#BSUB -R "select[model == XeonGold6226R]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot --
#BSUB -M 64GB
### -- set walltime limit: hh:mm --
#BSUB -W 48:00
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o Output_%J.out
#BSUB -e Error_%J.err

# here follow the commands you want to execute
module load gcc/12.2.0-binutils-2.39 ninja/1.10.2 cmake/3.26.4 boost/1.81.0-gcc-12.2.0
rm -rf cmake-build-release
#rm -f out_raw/*
export CC=gcc
export CGAL_DIR=~/Documents/cgal

cmake -G Ninja -DCMAKE_MAKE_PROGRAM=ninja -DCMAKE_CXX_FLAGS="-O3" -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release

./smalltest.sh
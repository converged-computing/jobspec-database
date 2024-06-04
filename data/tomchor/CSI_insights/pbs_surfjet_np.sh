#!/bin/bash -l
#PBS -A UMCP0012
#PBS -N PNN_CItest1_f8
#PBS -k eod
#PBS -o logs/PNN_CItest1_f8.out
#PBS -e logs/PNN_CItest1_f8.err
#PBS -l walltime=24:00:00
#PBS -q casper
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l gpu_type=v100
#PBS -M tchor@umd.edu
#PBS -m abe

# Clear the environment from any previously loaded modules
module purge
module load ncarenv/1.3 gnu/9.1.0 ncarcompilers/0.5.0
module load netcdf/4.7.4 openmpi/4.1.0 julia/1.6.0
module load peak_memusage
module li

#/glade/u/apps/ch/opt/usr/bin/dumpenv # Dumps environment (for debugging with CISL support)

export JULIA_DEPOT_PATH="/glade/work/tomasc/.julia_bkp"

peak_memusage.exe julia --project \
    surfjet_np.jl --simname=PNN_CItest1_f8 --factor=1 2>&1 | tee out/PNN_CItest1_f8.out


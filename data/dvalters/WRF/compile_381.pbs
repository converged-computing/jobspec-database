#!/bin/bash --login

#

#PBS -l select=serial=true:ncpus=1

#PBS -l walltime=04:00:00

#PBS -A n02-ncas

module load netcdf

export WRFIO_NCD_LARGE_FILE_SUPPORT=1

export NETCDF=/opt/cray/netcdf/default/cray/81

# Make sure any symbolic links are resolved to absolute path

export PBS_O_WORKDIR=$(readlink -f $PBS_O_WORKDIR)

# Change to the directory that the job was submitted from

cd $PBS_O_WORKDIR compile em_real >&compile_log_out.txt

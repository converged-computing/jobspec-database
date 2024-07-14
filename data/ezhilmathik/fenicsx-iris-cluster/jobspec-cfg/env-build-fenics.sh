#!/bin/bash -l
# User should set TAG and FLAGS appropriately
TAG=aion-master-r23 ### Iris has Broadwell and Skylake Intel architecture; adopt your software module according to 
export FLAGS="-O2 -march=znver2"
export MAKEFLAGS="-j16"

PREFIX=${HOME}/fenicsx-${TAG}
BUILD_DIR=/tmp/${USER}/fenicsx-${TAG}

module purge

if [[ $(hostname) =~ "iris" ]]; then
###    resif-load-swset-devel
    module use /opt/apps/resif/iris/2020b/broadwell/modules/all ### for example this is for Iris - Broadwell Intel architecture 
fi

# 2020b software set Intel MPI with GCC build
module load toolchain/iimpi/2020b
module load data/HDF5/1.10.7-iimpi-2020b

# 2020b software set OpenMPI with GCC build
#module load toolchain/gompi/2020b
#module load data/HDF5/1.10.7-gompi-2020b

module load devel/CMake/3.18.4-GCCcore-10.2.0
module load lang/flex/2.6.4-GCCcore-10.2.0
module load lang/Python/3.8.6-GCCcore-10.2.0
module load numlib/OpenBLAS/0.3.12-GCC-10.2.0
module load devel/Boost/1.74.0-GCC-10.2.0
module load lang/Bison/3.7.1-GCCcore-10.2.0

# Use GCC for non-MPI builds.
export CC=gcc
export CXX=g++
export FC=gfortran

# 2020b software set OpenMPI with GCC
#export MPICC=mpicc
#export MPICXX=mpicxx
#export MPIFC=mpif90

# 2020b software set Intel MPI with GCC
export MPICC=mpigcc
export MPICXX=mpigxx
export MPIFC=mpif90

export PATH=${PREFIX}/bin:${PATH}
export LD_LIBRARY_PATH=${PREFIX}/lib:${PREFIX}/lib64:${LD_LIBRARY_PATH}
export C_INCLUDE_PATH=${PREFIX}/include:${C_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=${PREFIX}/include:${CPLUS_INCLUDE_PATH}
export PKG_CONFIG_PATH=${PREFIX}/lib/pkgconfig:${PREFIX}/lib64/pkgconfig:$PKG_CONFIG_PATH

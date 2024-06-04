#!/bin/bash

###############################################################################
#
# This is an example build script for building Fluidity on Archer, and
# can be found in the GitHub FluidityProject/buildscripts repository
# as:
#
#  /uk/ac/archer/build.bash
# 
# along with the latest PrgEnv-fluidity module and compile and test
# scripts which will be required for this script to function properly.
#
# To check out the build scripts on Archer, start a clean login
# session, change to a directory on /home or /work, and then run:
#
#  git clone https://github.com/FluidityProject/buildscripts.git
#
# The resulting buildscripts/uk/ac/archer directory should be used for
# BUILDSCRIPTS below.
#
# To check out Fluidity on Archer, start a clean login session, change
# to a directory on /work, and then run:
#
#  git clone https://github.com/FluidityProject/fluidity.git
#
# Edit this script as directed in the comments below.  Change into the
# fluidity directory and run this script with:

#  <your own buildscripts directory>/build.bash
#
# The progress of the build can be monitored in timestamped files of
# the form:
#
#  configure-##########.log
#  compile-##########.log
#  unittest-##########.log
#  serialtest-##########.log
#  mediumtest-##########.log
#
###############################################################################

export TIMESTAMP=`date +%s`

# EDIT REQUIRED: set the project for running the PBS jobs
export PROJECT="y07"

# EDIT REQUIRED: Set the following to your own buildscripts directory
export BUILDSCRIPTS=/work/y07/y07/fluidity/buildscripts/uk/ac/archer

# EDIT REQUIRED: Set the following to your own Fluidity directory on
# /work
export FLUIDITYDIR=/work/y07/y07/fluidity/fluidity
 
# Copy the PrgEnv-fluidity module to the Fluidity directory so
# $BUILDSCRIPTS does not need to be accessed at all at run time
mkdir -p $FLUIDITYDIR/modulefiles
cp -p $BUILDSCRIPTS/PrgEnv-fluidity $FLUIDITYDIR/modulefiles
module use $FLUIDITYDIR/modulefiles

module unload PrgEnv-cray PrgEnv-gnu PrgEnv-intel
module load PrgEnv-fluidity

# PrgEnv-gnu is loaded by PrgEnv-fluidity
# python-compute is loaded by PrgEnv-fluidity
# pc-numpy is loaded by PrgEnv-fluidity

# Change to the Fluidity directory
cd $FLUIDITYDIR

export CPPFLAGS=""
export LDFLAGS=""
export LIBS=""

# Use only Zoltan from Trilinos.  This needs a check that the TPSL
# version is the same as for cray-petsc.  For cray-petsc/3.6.3.0 the
# correct version of cray-trilinos is 12.2.1.0 (TPSL 16.03 == TPSL
# 16.03.1)
#
# grep -rI 'TPSL' /opt/cray/petsc/[0-9]*/release_info | uniq
# /opt/cray/petsc/3.5.2.1/release_info:      TPSL 1.4.3
# /opt/cray/petsc/3.6.1.0/release_info:      TPSL 1.5.2
# /opt/cray/petsc/3.6.3.0/release_info:      TPSL 16.03
# /opt/cray/petsc/3.7.2.0/release_info:      TPSL 16.07.1
# /opt/cray/petsc/3.7.2.1/release_info:      TPSL 16.07.1
# /opt/cray/petsc/3.7.4.0/release_info:      TPSL 16.12.1
#
# grep -rI 'TPSL' /opt/cray/trilinos/[0-9]*/release_info | uniq
# /opt/cray/trilinos/11.12.1.0/release_info:      TPSL 1.4.3
# /opt/cray/trilinos/11.12.1.5/release_info:      TPSL 1.5.2
# /opt/cray/trilinos/12.2.1.0/release_info:      TPSL 16.03.1
# /opt/cray/trilinos/12.6.3.1/release_info:      TPSL 16.07.1
# /opt/cray/trilinos/12.6.3.2/release_info:      TPSL 16.07.1
# /opt/cray/trilinos/12.8.1.0/release_info:      TPSL 16.12.1
#
# If using the installed Zoltan causes problems (it shouldn't), then
# Zoltan will need to be built separately
module load cray-trilinos/12.10.1.0
CPPFLAGS="$CPPFLAGS -I$CRAY_TRILINOS_PREFIX_DIR/include"
LDFLAGS="$LDFLAGS -Wl,-rpath=$CRAY_TRILINOS_PREFIX_DIR/lib"
LIBS="$LIBS -L$CRAY_TRILINOS_PREFIX_DIR/lib -lzoltan"
LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CRAY_TRILINOS_PREFIX_DIR/lib"
module unload cray-trilinos

#module load cray-hdf5
module load cray-netcdf-hdf5parallel
module load cray-hdf5-parallel/1.10.2.0
# cray-petsc/3.6.3.0 needs cray-mpich >= 7.3 (all the other cray-petsc just need cray-mpich >= 7.0)
# cray-petsc/3.6.3.0 needs cray-tpsl >= 16.03.1 (and has been built with 16.03 == 16.03.1)
#module switch cray-mpich/7.3.2
module load cray-tpsl/17.04.1
module load cray-petsc/3.7.5.0
module load cmake
module load boost
module load vtk

module list > configure-${TIMESTAMP}.log 2>&1

export PATH=$PATH:$FLUIDITYDIR/bin
export PYTHONPATH=$PYTHONPATH:$FLUIDITYDIR/python

# Tell Fluidity that we're on ARCHER; used by configure
export ARCHER="yes"

# Compile dynamic loaded libraries and fix the library paths to search
export CRAYPE_LINK_TYPE="dynamic" 
export CRAY_ADD_RPATH="yes"

CPPFLAGS="$CPPFLAGS -DMPICH_IGNORE_CXX_SEEK"

export CC="cc"
export MPICC="cc"
export CXX="CC"
export MPICXX="CC"
export FC="ftn"
export MPIFC="ftn"
export F77="ftn"
export MPIF77="ftn"
export F90="ftn"
export MPIF90="ftn"

echo "CPPFLAGS=$CPPFLAGS" >> configure-${TIMESTAMP}.log
echo "LIBS=$LIBS" >> configure-${TIMESTAMP}.log

# A bit of prep for h5hut
libtoolize

# The configure step needs to run on a login node (which is almost
# identical to a compute node) and the compilation should be run on a
# PP node because it takes a long time.  compile.pbs has "#PBS -V", so
# it will get all the environment set up by this script.

if ./configure --enable-2d-adaptivity >> configure-${TIMESTAMP}.log 2>&1; then
    job1=$(qsub -A $PROJECT $BUILDSCRIPTS/compile.pbs)
else
    echo "configure_failed"
    exit 1
fi

# Test aren't working yet
#
# Tests are meant to test the run time setup, so only pass the
# time-stamp and the Fluidity directory.
#qsub -W depend=afterok:$job1 -A $PROJECT -v TIMESTAMP,FLUIDITYDIR $BUILDSCRIPTS/unittest.pbs
#qsub -W depend=afterok:$job1 -A $PROJECT -v TIMESTAMP,FLUIDITYDIR $BUILDSCRIPTS/serialtest.pbs
#qsub -W depend=afterok:$job1 -A $PROJECT -v TIMESTAMP,FLUIDITYDIR $BUILDSCRIPTS/mediumtest.pbs

exit 0

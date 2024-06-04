#!/bin/bash --login

###############################################################################
#
# This is an example PBS script for building Fluidity on Archer, and can be
# found in the GitHub FluidityProject/buildscripts repository as:
#
#     /uk/ac/archer/compile.pbs
#
# This script is submitted by build.bash to compile Fluidity using the
# configuration set up by build.bash
#
# The progress of the compilation can be monitored in a timestamped
# file of the form:
#
#  compile-##########.log
#
###############################################################################

#PBS -N compile
#PBS -l select=serial=true:ncpus=8
#PBS -l walltime=1:00:00
#PBS -V

# Make sure any symbolic links are resolved to absolute path
export PBS_O_WORKDIR=$(readlink -f $PBS_O_WORKDIR)

# Change to the directory that the job was submitted from
cd $PBS_O_WORKDIR

# The configuration has to be done already, on the login nodes, and
# the PBS -V directive (see above) gets the environment that has been
# set up.  The compilation takes about 30 minutes ('make -j 8') on a
# serial node.

# The compile step is run on the serial nodes because the compile
# takes so long, and some optimisation steps may take so long that the
# /tmp directory is emptied by the system during the optimisations,
# giving 'file not found' errors.  This seems to be occurring on the
# serial nodes as well now so TMPDIR is set; this may affect Cray
# Fortran OPEN statements for scratch files.
mkdir -p tmp
export TMPDIR=$PWD/tmp

make clean > compile-${TIMESTAMP}.log 2>&1
# The j value MUST be the same as the ncpus value in the #PBS -l
# select line.
make -j 8 >> compile-${TIMESTAMP}.log 2>&1
make mp >> compile-${TIMESTAMP}.log 2>&1
#make -j 8 fltools >> compile-${TIMESTAMP}.log 2>&1
#make -j 8 build_unittest >> compile-${TIMESTAMP}.log 2>&1

unset TMPDIR
rm -rf tmp

#!/bin/bash --login

###################################################################
#
# This is an example PBS script for archer, and can be found in the
# github FluidityProject/buildscripts repository as:
#
#     /uk/ac/archer/top_hat.pbs
#
# That directory also contains a build script for Fluidity on Archer,
# along with the latest PrgEnv-fluidity module which will need to be
# present as described below for this script to function.
#
# It is intended to be a drop-in script for the examples/top_hat/
# directory to preprocess and run the cg, dg, and cv models.
#
# Edit as directed in the comments below.
#
###################################################################

#PBS -l walltime=00:05:00
#PBS -N top_hat

# NOTE: Archer is entirely composed of 24-core nodes; 'select' gives
#       you multiples of 24 cores, and there is no possibility to
#       allocate partial nodes

#PBS -l select=1

# EDIT REQUIRED: set the project this job should be billed to 
#PBS -A z19-cse

# EDIT REQUIRED: Set the following to your own Fluidity install
# directory
export FLUIDITYDIR=/work/z01/z01/mjf/FluidityProject/fluidity

# Make sure any symbolic links are resolved to absolute path
export PBS_O_WORKDIR=$(readlink -f $PBS_O_WORKDIR)

# Change to the directory that the job was submitted from
cd $PBS_O_WORKDIR

module use $FLUIDITYDIR/modulefiles
module unload PrgEnv-cray PrgEnv-gnu PrgEnv-intel
module load PrgEnv-fluidity
export PATH=$PATH:$FLUIDITYDIR/bin
export PYTHONPATH=$PYTHONPATH:$FLUIDITYDIR/python

# NOTE: To use hyperthreading, pass '-j 2' to aprun

time aprun -n 1 interval --dx=0.025 --reverse 0 3 line

echo **********Running the Continuous Galerkin version of this example:
time aprun -n 1 fluidity -v2 -l $FLUIDITYDIR/examples/top_hat/top_hat_cg.flml
echo **********Running the Discontinuous Galerkin version of this example:
time aprun -n 1 fluidity -v2 -l $FLUIDITYDIR/examples/top_hat/top_hat_dg.flml
echo **********Running the Control Volumes version of this example:
time aprun -n 1 fluidity -v2 -l $FLUIDITYDIR/examples/top_hat/top_hat_cv.flml

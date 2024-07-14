#!/bin/bash

##
## @file run_grid.sh
##
## Description of the script file
##
## @author cuto <Jakub.Cuth@cern.ch>
## @date 2015-10-27

# read job number
jobseed=$1
shift
# read the rest
arguments=$*

# get random seed from config file (xargs will remove leading and trailing spaces)
configseed=`grep rseed input.in | cut -d= -f2 | cut -d\# -f1 | xargs`
jobseed=$((jobseed+configseed))
# add jobseed
arguments="${arguments} --seed ${jobseed}"

# grid verbosity
arguments="${arguments} --grid"

# setup ENV
source ./setup.sh

# LHAPDF
if [[ SETTARGET =~ compile ]]
then
    # setup sft
    lsetup "sft releases/MCGenerators/lhapdf/6.1.5-2f446"
    lhapdf-config --version || exit 2
    # official sets
    export LHAPDF_DATA_PATH=/cvmfs/sft.cern.ch/lcg/external/lhapdfsets/current/
else
    # own lhapdf set
    export LHAPDF_DATA_PATH=$LHAPDF_DATA_PATH:./LHAPDF
fi
#export LHAPATH=$LHAPDF_DATA_PATH
# dyturbo libs
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./lib
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/cvmfs/sft.cern.ch/lcg/external/gcc/4.8.1/x86_64-slc6-gcc48-opt/lib/../lib64

echo
echo LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
echo

# compile just for sure
#make install

echo
echo input.in :
cat input.in
echo
rm -f results*.root

./bin/dyturbo $arguments



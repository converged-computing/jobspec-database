#!/bin/bash

##
## @file compile_grid.sh
##
## Description of the script file
##
## @author cuto <Jakub.Cuth@cern.ch>
## @date 2015-11-11

DYTURBOVERSION=dyturbo-$1

tar zxvf ${DYTURBOVERSION}.tar.gz
cd ${DYTURBOVERSION}

# setup ENV
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
# LHAPDF and ROOT
#lsetup "root 6.04.06-x86_64-slc6-gcc48-opt"
#lsetup "sft releases/MCGenerators/lhapdf/6.1.5-6570e" 
# --rootVer=6.04/14 --cmtConfig=x86_64-slc6-gcc49-opt
lsetup "root 6.04.14-x86_64-slc6-gcc49-opt"
lsetup "sft releases/MCGenerators/lhapdf/6.1.5-2f446"

lhapdf-config --version || exit 2

# own lhapdf set
export LHAPDF_DATA_PATH=./
# official sets
export LHAPDF_DATA_PATH=/cvmfs/sft.cern.ch/lcg/external/lhapdfsets/current
export LHAPATH=$LHAPDF_DATA_PATH

mkdir -p m4 || exit 1
echo install Cuba
./install-cuba || exit 5
echo install DYTURBO
#autoreconf -i || exit 2
#cp ../quadrules.f src/.
./configure --enable-Ofast --enable-root || exit 3
make && make install || exit 4

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./lib
export LD_LIBRARY_PATH=/cvmfs/sft.cern.ch/lcg/external/gcc/4.8.1/x86_64-slc6-gcc48-opt/lib/../lib64:${LD_LIBRARY_PATH}

#install
cd ../
cp -R ${DYTURBOVERSION}/bin .
cp -R ${DYTURBOVERSION}/lib .


ls -la bin/dyturbo || exit 6

echo Compilation successfull


exit 0

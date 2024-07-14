export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh " "
## gcc ROOT
lsetup "gcc gcc493_x86_64_slc6"
lsetup "root 6.04.14-x86_64-slc6-gcc49-opt"

#lsetup "root 6.14.08-x86_64-centos7-gcc8-opt"

#setup recent gcc version and root
#. /cvmfs/sft.cern.ch/lcg/contrib/gcc/4.9.3/x86_64-cc7-gcc49-opt/setup.sh
#. /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.06.08/x86_64-centos7-gcc49-opt/root/bin/thisroot.sh

#. /cvmfs/sft.cern.ch/lcg/releases/gcc/8.2.0-3fa06/x86_64-centos7/setup.sh
#. /cvmfs/sft.cern.ch/lcg/releases/LCG_94a/ROOT/6.14.08/x86_64-centos7-gcc8-opt/bin/thisroot.sh

#--rootVer=6.14/08 --cmtConfig=x86_64-centos7-gcc8-opt

#export PATH=/cvmfs/sft.cern.ch/lcg/releases/MCGenerators/lhapdf/6.2.3-7c75b/x86_64-centos7-gcc62-opt/bin:$PATH

#export PATH=/cvmfs/sft.cern.ch/lcg/releases/MCGenerators/lhapdf/6.2.3-7c75b/x86_64-centos7-gcc8-opt/bin:$PATH

export PATH=/cvmfs/sft.cern.ch/lcg/releases/MCGenerators/lhapdf/6.2.3-0ff65/x86_64-slc6-gcc49-opt/bin:$PATH
export LHAPDF_DATA_PATH=/cvmfs/sft.cern.ch/lcg/external/lhapdfsets/current/
export LHAPATH=/cvmfs/sft.cern.ch/lcg/external/lhapdfsets/current/

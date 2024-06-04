#!/bin/sh -xe
#SBATCH --job-name=FV3_container
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --time=00:15:00               # Time limit hrs:min:sec
#SBATCH --output=FV3_container_%j.log   # Standard output and error log

##############################################################################
## User set up veriables
## Root directory for CI
dirRoot=/contrib/fv3
## Intel version to be used
intelVersion=2022.1.1
##############################################################################
## HPC-ME container
container=/contrib/containers/HPC-ME_base-ubuntu20.04-intel${intelVersion}.sif 
## Set up the directories
if [ -z "$1" ]
  then
    echo "No branch supplied; using main"
    branch=main
  else
    echo Branch is ${1}
    branch=${1}
fi

testRoot=${dirRoot}/${intelVersion}/${branch}
buildDir=${testRoot}/build
srcDir=${testRoot}/src
logDir=${testRoot}/log
## create directories
rm -rf ${testDir}
mkdir -p ${buildDir}
mkdir -p ${logDir}
mkdir -p ${srcDir}
## Install singularity
sudo yum install -y singularity
## clone code
cd ${srcDir}

## Check out the PR
# example below on how to do it for FMS
#cd FMS && git fetch origin ${1}:toMerge && git merge toMerge |& tee ${logdir}/fetch.log

# Set up build
cd ${buildDir}
#Build FMS
set -o pipefail
singularity exec -B /lustre,/contrib ${container} hostname |& tee ${logDir}/compile.log # INSERT BUILD SCRIPT FOR hostname ;  PIPE OUTPUT TO ${logDir}/compile.log
# Run the build
set -o pipefail
singularity exec -B /lustre,/contrib ${container} hostname |& tee ${logDir}/run.log # INSERT RUN SCRIPT FOR hostname ; PIPE OUTPUT TO ${logDir}/run.log

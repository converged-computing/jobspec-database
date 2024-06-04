#!/bin/sh -xe
#SBATCH --job-name=FMS_container
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --time=00:15:00               # Time limit hrs:min:sec
#SBATCH --output=FMS_container_%j.log   # Standard output and error log
intelVersion=2021.2
container=/contrib/intel${intelVersion}_netcdfc4.7.4_ubuntu.sif

buildDir=/contrib/${intelVersion}/${1}
logdir=/contrib/${intelVersion}/${1}/log
# create directories

rm -rf ${buildDir}
mkdir -p ${buildDir}/build
mkdir -p ${logdir}

#Install singularity
sudo yum install -y singularity

#clone FMS
cd ${buildDir}
rm -rf FMS/
git clone https://github.com/NOAA-GFDL/FMS.git |& tee ${logdir}/clone.log
# Check out the PR
cd FMS && git fetch origin ${1}:toMerge && git merge toMerge |& tee ${logdir}/fetch.log

# Set up build
cd ${buildDir}/build
#Build FMS
set -o pipefail
singularity exec -B /lustre,/contrib ${container} autoreconf -i ${buildDir}/FMS/configure.ac |& tee ${logdir}/contreconf.log
set -o pipefail
singularity exec -B /lustre,/contrib ${container} ${buildDir}/FMS/configure --prefix=${buildDir}/build FC=mpiifort CC="mpiicc -no-multibyte-chars" CPPFLAGS="-L/opt/netcdf-fortran/lib -lnetcdff -I/opt/netcdf-fortran/include -I/opt/netcdf-fortran/include -I/opt/netcdf-c/include -I/opt/hdf5/include -I/include -L/opt/netcdf-c/lib -lnetcdf" LIBS="-L/opt/netcdf-fortran/lib -lnetcdff -L/opt/netcdf-c/lib -lnetcdf" |& tee ${logdir}/configrun.log
cp config.log ${logdir}/config.log 
# Run the build
set -o pipefail
singularity exec -B /lustre,/contrib ${container} /contrib/make_fms_container_pr.sh ${buildDir}/build |& tee ${logdir}/make_fms_container.log

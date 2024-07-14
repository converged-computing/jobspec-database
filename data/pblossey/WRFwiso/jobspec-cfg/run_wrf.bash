#!/bin/bash
#PBS -N AntarcWISO_wrf_1853-01
#PBS -A UWAS0052
#PBS -l walltime=12:00:00
#PBS -q economy
#PBS -j oe
#PBS -m e
#PBS -M pblossey@uw.edu
#PBS -l select=4:ncpus=36:mpiprocs=36

DATETAG=1853-01

BASEDIR=/glade/u/home/pblossey/scratch/WAIS/TopoiCESM/TopoPI_OceanPI/WRFwiso_TopoPI_OceanPI_${DATETAG}/WRFV3/test/em_real/

TODAY=`date --iso-8601`

### Set TMPDIR as recommended
export TMPDIR=/glade/scratch/$USER/temp
mkdir -p $TMPDIR

cd $BASEDIR

## load modules for intel/netcdf setup -- swap to Intel 18.0.5
#source /etc/profile.d/modules.sh
#module unload intel netcdf mpt
#module load intel/18.0.5 mpt/2.18 netcdf/4.6.1
#module list

### Run the executable
mpiexec_mpt dplace -s 1 ./wrf.exe > log_wrf_${TODAY}

popd

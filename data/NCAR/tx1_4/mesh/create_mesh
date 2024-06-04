#!/bin/bash
#PBS -A P93300012
#PBS -N esmf_mesh
#PBS -j oe
#PBS -q regular
#PBS -l walltime=01:00:00
#PBS -l select=1:mpiprocs=36
#PBS -o log.out

#export TMPDIR=/glade/scratch/$USER/temp
#mkdir -p $TMPDIR
#export MPI_USE_ARRAY=false

# load modules
module purge
module load ncarenv/1.3
module load intel/19.0.2
module load mpt/2.19
module load netcdf-mpi/4.7.1
module load pnetcdf/1.11.0
module load ncarcompilers/0.5.0
module use /glade/work/turuncu/PROGS/modulefiles/esmfpkgs/intel/19.0.2
module load esmf-8.0.0-ncdfio-mpt-O


file_i="tx1_4_SCRIP_221216.nc"
file_o="tx1_4_mesh_221216.nc"
ESMF_exe=/glade/work/turuncu/PROGS/esmf/8.0.0/mpt/2.19/intel/19.0.2/bin/binO/Linux.intel.64.mpt.default/ESMF_Scrip2Unstruct

mpiexec_mpt -np 36 $ESMF_exe $file_i $file_o  0 ESMF


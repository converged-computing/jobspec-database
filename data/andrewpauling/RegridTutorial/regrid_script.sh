#!/bin/bash
# Batch system directives
#PBS  -A UWAS0064
#PBS  -N regrid_script
#PBS  -q economy
#PBS  -l walltime=06:00:00
#PBS  -j oe
#PBS  -S /bin/bash
#PBS  -l select=10:ncpus=36:mpiprocs=36
#PBS -m ea
#PBS -M apauling@uw.edu

module load esmf_libs
module load esmf-8.0.0-ncdfio-mpi-g

echo "Waiting for regridding to start"

mpiexec_mpt ESMF_RegridWeightGen --ignore_unmapped --netcdf4 --weight_only -s ice5g_anomaly_remap_1.9x2.5.nc -d gmted2010_modis-4regrid.nc -w weights_2deg_to_WRF.nc -m bilinear
wait

echo "Regridding Finished"

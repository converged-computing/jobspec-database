#!/bin/bash
#PBS  -N MODEL_aws_CYC_prod
#PBS  -A NOSOFS-DEV
#PBS  -q dev_transfer
#PBS  -o /lfs/h1/nos/PLATFORM/LOGNAME1/rpt/NOSOFSVER/MODEL_aws_CYC_prod.out
#PBS  -e /lfs/h1/nos/PLATFORM/LOGNAME1/rpt/NOSOFSVER/MODEL_aws_CYC_prod.err
#PBS  -l place=vscatter,select=1:ncpus=8
#PBS  -l walltime=1:30:00
##export NODES=1
##export LSB_MAX_NUM_PROCESSORS=$(($NODES*128))

. VERSION_FILE

RPTDIR=/lfs/h1/nos/PLATFORM/$LOGNAME/rpt/NOSOFSVER
WORKDIR=/lfs/h1/nos/PLATFORM/$LOGNAME/work/${nosofs_ver}/MODEL 
if [ ! -r $WORKDIR ]; then
  mkdir -p -m 755 $WORKDIR
fi 
cd ${WORKDIR}

module purge
module load envvar/$envvars_ver

# Loading Intel Compiler Suite
module load PrgEnv-intel/${PrgEnv_intel_ver}
module load craype/${craype_ver}
module load intel/${intel_ver}
#module load cray-mpich/${cray_mpich_ver}
module load cray-pals/${cray_pals_ver}
#Set other library variables
module load libjpeg/${libjpeg_ver}
module load netcdf/${netcdf_ver}
module load hdf5/${hdf5_ver}
module load subversion/${subversion_ver}
module load python/${python_ver}
module load prod_envir/${prod_envir_ver}
module load prod_util/${prod_util_ver}
module load grib_util/${grib_util_ver}
module load cfp/${cfp_ver} 
#module load nco/${nco_ver}
module load awscli/1.16.308

# EXPORT list here

set +x
export envir=prod
export OFS=MODEL
export cyc=CYC
export job=MODEL_aws_CYC
export platform=PLATFORM

####   For testing or development run
#export PDY=20210824
export KEEPDATA=YES
export SENDCOM=NO
export SENDDBN=NO
export SENDSMS=NO
export PACKAGEROOT=/lfs/h1/nos/nosofs/noscrub/$LOGNAME/packages
export COMPATH=/lfs/h1/ops/prod/com
#export COMROOT=/lfs/h1/ops/prod/com
export DCOMROOT=/lfs/h1/ops/prod/dcom
export DATAROOT=/lfs/h1/nos/PLATFORM/$LOGNAME/work/${nosofs_ver}/${OFS}
#####################################################

# CALL executable job script here
export pbsid=${PBS_JOBID%%.*}

JOB_SCRIPT_FILETRANSFER

#s=${PBS_JOBID}
#mv ${RPTDIR}/MODEL_aws_CYC_prod.out ${RPTDIR}/MODEL_aws_CYC_prod.out."${s%%.*}"
#mv ${RPTDIR}/MODEL_aws_CYC_prod.err ${RPTDIR}/MODEL_aws_CYC_prod.err."${s%%.*}"
#######################################################################
# Purpose: To execute the job that transfer the NOS OFS files from wcoss onto CO-OPS aws s3 
#          on the IBMSP
#######################################################################


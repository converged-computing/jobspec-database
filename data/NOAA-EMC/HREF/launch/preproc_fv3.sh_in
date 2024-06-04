#! /bin/sh

#PBS -j oe
#PBS -o /lfs/h2/emc/lam/noscrub/Matthew.Pyle/rrfs.v3.1.0/launch/logs/preproc_fv3.out__DOM__CYC_
#PBS -e /lfs/h2/emc/lam/noscrub/Matthew.Pyle/rrfs.v3.1.0/launch/logs/preproc_fv3.err__DOM__CYC_
#PBS -l select=_NODES_:ncpus=_PTILE_:mem=256GB -l place=excl
#PBS -q dev
#PBS -A HREF-DEV
#PBS -l walltime=000:27:05

export HOMEhref=/lfs/h2/emc/lam/noscrub/${USER}/rrfs.v3.1.0
source ${HOMEhref}/versions/run.ver

export OMP_NUM_THREADS=1

module purge
module load envvar/1.0
module load PrgEnv-intel/8.1.0
module load craype/2.7.13
module load intel/19.1.3.304
module load cray-mpich/8.1.12
module load cray-pals/1.0.12

module load prod_util/${prod_util_ver}
module load wgrib2/${wgrib2_ver}
module load netcdf/${netcdf_ver}
module load g2/${g2_ver}
module load g2tmpl/${g2tmpl_ver}
module load jasper/${jasper_ver}
module load libpng/${libpng_ver}
module load zlib/${zlib_ver}
module load cfp/${cfp_ver}

export numprocs=_NTASK_
export spanprocs=_PTILE_
export NTASK=_NTASK_
export PTILE=_PTILE_

export RUN_ENVIR=test
export envir=test
export NET=rrfs
export RUN=rrfs

export NEST=_DOM_
export cyc=_CYC_
export DATE=_DATE_

export PDY=${DATE}
export PDYm1=`$NDATE -24 ${PDY}${cyc} | cut -c1-8`

echo PDY is $PDY
export PDYm1 is $PDYm1

export DATAROOT=/lfs/h2/emc/stmp/${USER}/tmp
export COMROOT=/lfs/h2/emc/ptmp/${USER}/${envir}/com
export COMPATH=/lfs/h2/emc/ptmp/${USER}

export DATA=${DATAROOT}/href_preprocfv3_${NEST}_${cyc}_${envir}

echo DATA is $DATA

# export COMINnam=`compath.py -e canned nam/${nam_ver}/nam`
# export COMINrrfs=`compath.py -e canned rrfs/${rrfs_ver}/rrfs`
export COMINrrfs="/lfs/h2/emc/lam/noscrub/Matthew.Pyle/com/rrfs/rrfs"
export GESROOT=`compath.py -o ${NET}/${href_ver}/nwges`

if [ ! -e $GESROOT ] ; then
 mkdir -p $GESROOT
fi

export KEEPDATA=NO

export SENDCOM=YES
export SENDECF=NO
export SENDDBN=NO
export job=href_preproc_fv3_${NEST}_${cyc}

$HOMEhref/jobs/JHREF_PREPROC_FV3

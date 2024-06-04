#!/bin/bash

#---------------------------------------------------------------------
# Driver script for running on Hera.
#
# Edit the 'config' file before running.
#---------------------------------------------------------------------

set -x

compiler=intel
target=hera
ufsutil_dir=/scratch1/BMC/gsd-fv3-dev/MAPP_2018/bhuang/JEDI-2020/JEDI-FV3/expCodes/GSDChem_cycling/global-workflow-CCPP2-Chem/gsd-ccpp-chem/sorc/UFS_UTILS_20221207/UFS_UTILS
source ${ufsutil_dir}/sorc/machine-setup.sh > /dev/null 2>&1
module use ${ufsutil_dir}/modulefiles
module load build.$target.$compiler
module list

# Needed for NDATE utility
module use -a /scratch2/NCEPDEV/nwprod/NCEPLIBS/modulefiles
module load prod_util/1.1.0
module load hpss

NDATE=/scratch2/NCEPDEV/nwprod/NCEPLIBS/utils/prod_util.v1.1.0/exec/ndate

CURDIR=`pwd`
DRIVER=${CURDIR}/driver.hera_hpssDownload_v15.sh
DATES=2020052500 #2019061200
DATEE=2020063018 #2021032018
gfs_ver=v15
PROJECT_CODE=wrf-chem
QUEUE=batch
UFS_DIR=${ufsutil_dir}
CDUMP=gdas
EXTRACT_DATA=yes
RUN_CHGRES=no
TMPDIR=/scratch1/BMC/chem-var/MAPP_2018/bhuang/BackupGdas/
HPSSDIR=/BMC/fim/5year/MAPP_2018/bhuang/BackupGdas
use_v16retro=no

CYCLE_LOG=${CURDIR}/CYCLE.info
HPSS_EXTRACT_LOG=${CURDIR}/record.extract_success_${gfs_ver}
HPSS_HTAR_LOG=${CURDIR}/record.hpss_htar_success_${gfs_ver}

cdate=`cat ${CYCLE_LOG}`
cdatem6=`${NDATE} -6 ${cdate}`
export yy=`echo ${cdate} | cut -c 1-4`
export mm=`echo ${cdate} | cut -c 5-6`
export dd=`echo ${cdate} | cut -c 7-8`
export hh=`echo ${cdate} | cut -c 9-10`

if [ ${cdate} -lt ${DATES} ] ||  [ ${cdate} -gt ${DATEE} ]; then
    echo "Error: Current date is not within ${DATES} and ${DATEE} and exit."
    exit 1
fi

#if [ "$use_v16retro" = "yes" ]; then
#
#  gfs_ver=v16retro
#
#else
#
#  gfs_ver=v16
#
# No ENKF data prior to 2012/05/21/00z
#  if [ $yy$mm$dd$hh -lt 2012052100 ]; then
#    set +x
#    echo FATAL ERROR: SCRIPTS DO NOT SUPPORT OLD GFS DATA
#    exit 2
#  elif [ $yy$mm$dd$hh -lt 2016051000 ]; then
#    gfs_ver=v12
#  elif [ $yy$mm$dd$hh -lt 2017072000 ]; then
#    gfs_ver=v13
#  elif [ $yy$mm$dd$hh -lt 2019061200 ]; then
#    gfs_ver=v14
#  elif [ $yy$mm$dd$hh -lt 2021032100 ]; then
#    gfs_ver=v15
## The way the v16 switch over was done, there is no complete
## set of v16 or v15 data for 2021032100.  And although
## v16 was officially implemented 2021032212, the v16 prod 
## tarballs were archived starting 2021032106.
##  elif [ $yy$mm$dd$hh -lt 2021032106 ]; then
#    set +x
#    echo FATAL ERROR: NO V15 OR V16 DATA FOR 2021032100
#    exit 1
#  fi
#
#fi

if [ ${cdate} -eq ${DATES} ]; then
    echo "HPSS_EXTRACT at ${cdate}"
elif [ -f ${HPSS_HTAR_LOG} ]; then
    lastcyc=`tail -n 1 ${HPSS_HTAR_LOG}`
    if [ ${lastcyc} -eq ${cdatem6} ]; then
        echo "HPSS_EXTRACT at ${cdate}"
    else
        echo "Cycle at ${cdatem6} is not finished yet and wait..."
        exit 0
    fi
else
    echo "Current date not equal to strting date, nor does HPSS_HTAR_LOG file exist. Reset current date and exit now"
    exit 1
fi

EXTRACT_DIR=${TMPDIR}/${gfs_ver}/${cdate}

export EXTRACT_DIR gfs_ver yy mm dd hh 

MEM=6000M
WALLT="4:00:00"

if [ $EXTRACT_DATA == yes ]; then

  rm -fr $EXTRACT_DIR
  mkdir -p $EXTRACT_DIR


  case $gfs_ver in
    v14)
      DATAH=$(sbatch --parsable --partition=service --ntasks=1 --mem=$MEM -t $WALLT -A $PROJECT_CODE -q $QUEUE -J get_${CDUMP} \
       -o log.data.${CDUMP} -e log.data.${CDUMP} ${CURDIR}/get_v14.data.sh ${CDUMP})
      DEPEND="-d afterok:$DATAH"
      if [ "$CDUMP" = "gdas" ] ; then
        DATA1=$(sbatch --parsable --partition=service --ntasks=1 --mem=$MEM -t $WALLT -A $PROJECT_CODE -q $QUEUE -J get_enkf \
         -o log.data.enkf -e log.data.enkf ${CURDIR}/get_v14.data.sh enkf)
        DEPEND="-d afterok:$DATAH:$DATA1"
      fi
      ;;
    v15)
      DATAH=$(sbatch --parsable --partition=service --ntasks=1 --mem=$MEM -t $WALLT -A $PROJECT_CODE -q $QUEUE -J get_${CDUMP} \
       -o ${EXTRACT_DIR}/log.data.${cdate}.${CDUMP} -e ${EXTRACT_DIR}/log.data.${cdate}.${CDUMP} ${CURDIR}/get_v15.gdas.sh ${CDUMP})
      DATA1=$(sbatch --parsable --partition=service --ntasks=1 --mem=$MEM -t $WALLT -A $PROJECT_CODE -q $QUEUE -J get_grp1 \
       -o ${EXTRACT_DIR}/log.data.${cdate}.grp1 -e ${EXTRACT_DIR}/log.data.${cdate}.grp1 ${CURDIR}/get_v15.gdas.sh grp1)
      DATA2=$(sbatch --parsable --partition=service --ntasks=1 --mem=$MEM -t $WALLT -A $PROJECT_CODE -q $QUEUE -J get_grp2 \
       -o ${EXTRACT_DIR}/log.data.${cdate}.grp2 -e ${EXTRACT_DIR}/log.data.${cdate}.grp2 ${CURDIR}/get_v15.gdas.sh grp2)
      DATA3=$(sbatch --parsable --partition=service --ntasks=1 --mem=$MEM -t $WALLT -A $PROJECT_CODE -q $QUEUE -J get_grp3 \
       -o ${EXTRACT_DIR}/log.data.${cdate}.grp3 -e ${EXTRACT_DIR}/log.data.${cdate}.grp3 ${CURDIR}/get_v15.gdas.sh grp3)
      DATA4=$(sbatch --parsable --partition=service --ntasks=1 --mem=$MEM -t $WALLT -A $PROJECT_CODE -q $QUEUE -J get_grp4 \
       -o ${EXTRACT_DIR}/log.data.${cdate}.grp4 -e ${EXTRACT_DIR}/log.data.${cdate}.grp4 ${CURDIR}/get_v15.gdas.sh grp4)
      DATA5=$(sbatch --parsable --partition=service --ntasks=1 --mem=$MEM -t $WALLT -A $PROJECT_CODE -q $QUEUE -J get_grp5 \
       -o ${EXTRACT_DIR}/log.data.${cdate}.grp5 -e ${EXTRACT_DIR}/log.data.${cdate}.grp5 ${CURDIR}/get_v15.gdas.sh grp5)
      DATA6=$(sbatch --parsable --partition=service --ntasks=1 --mem=$MEM -t $WALLT -A $PROJECT_CODE -q $QUEUE -J get_grp6 \
       -o ${EXTRACT_DIR}/log.data.${cdate}.grp6 -e ${EXTRACT_DIR}/log.data.${cdate}.grp6 ${CURDIR}/get_v15.gdas.sh grp6)
      DATA7=$(sbatch --parsable --partition=service --ntasks=1 --mem=$MEM -t $WALLT -A $PROJECT_CODE -q $QUEUE -J get_grp7 \
       -o ${EXTRACT_DIR}/log.data.${cdate}.grp7 -e ${EXTRACT_DIR}/log.data.${cdate}.grp7 ${CURDIR}/get_v15.gdas.sh grp7)
      DATA8=$(sbatch --parsable --partition=service --ntasks=1 --mem=$MEM -t $WALLT -A $PROJECT_CODE -q $QUEUE -J get_grp8 \
       -o ${EXTRACT_DIR}/log.data.${cdate}.grp8 -e ${EXTRACT_DIR}/log.data.${cdate}.grp8 ${CURDIR}/get_v15.gdas.sh grp8)
      DEPEND="-d afterok:$DATAH:$DATA1:$DATA2:$DATA3:$DATA4:$DATA5:$DATA6:$DATA7:$DATA8"
      ;;
    v16retro)
      DATAH=$(sbatch --parsable --partition=service --ntasks=1 --mem=$MEM -t $WALLT -A $PROJECT_CODE -q $QUEUE -J get_v16retro \
       -o log.data.v16retro -e log.data.v16retro ./get_v16retro.data.sh ${CDUMP})
      DEPEND="-d afterok:$DATAH"
      ;;
    v16)
      DATAH=$(sbatch --parsable --partition=service --ntasks=1 --mem=$MEM -t $WALLT -A $PROJECT_CODE -q $QUEUE -J get_${CDUMP} \
       -o log.data.${CDUMP} -e log.data.${CDUMP} ./get_v16.data.hpssDownload.sh ${CDUMP})
      DEPEND="-d afterok:$DATAH"
      if [ "$CDUMP" = "gdas" ] ; then
        DATA1=$(sbatch --parsable --partition=service --ntasks=1 --mem=$MEM -t $WALLT -A $PROJECT_CODE -q $QUEUE -J get_grp1 \
         -o log.data.grp1 -e log.data.grp1 ./get_v16.data.hpssDownload.sh grp1)
        DATA2=$(sbatch --parsable --partition=service --ntasks=1 --mem=$MEM -t $WALLT -A $PROJECT_CODE -q $QUEUE -J get_grp2 \
         -o log.data.grp2 -e log.data.grp2 ./get_v16.data.hpssDownload.sh grp2)
        DATA3=$(sbatch --parsable --partition=service --ntasks=1 --mem=$MEM -t $WALLT -A $PROJECT_CODE -q $QUEUE -J get_grp3 \
         -o log.data.grp3 -e log.data.grp3 ./get_v16.data.sh grp3)
        DATA4=$(sbatch --parsable --partition=service --ntasks=1 --mem=$MEM -t $WALLT -A $PROJECT_CODE -q $QUEUE -J get_grp4 \
         -o log.data.grp4 -e log.data.grp4 ./get_v16.data.sh grp4)
        DATA5=$(sbatch --parsable --partition=service --ntasks=1 --mem=$MEM -t $WALLT -A $PROJECT_CODE -q $QUEUE -J get_grp5 \
         -o log.data.grp5 -e log.data.grp5 ./get_v16.data.sh grp5)
        DATA6=$(sbatch --parsable --partition=service --ntasks=1 --mem=$MEM -t $WALLT -A $PROJECT_CODE -q $QUEUE -J get_grp6 \
         -o log.data.grp6 -e log.data.grp6 ./get_v16.data.sh grp6)
        DATA7=$(sbatch --parsable --partition=service --ntasks=1 --mem=$MEM -t $WALLT -A $PROJECT_CODE -q $QUEUE -J get_grp7 \
         -o log.data.grp7 -e log.data.grp7 ./get_v16.data.sh grp7)
        DATA8=$(sbatch --parsable --partition=service --ntasks=1 --mem=$MEM -t $WALLT -A $PROJECT_CODE -q $QUEUE -J get_grp8 \
         -o log.data.grp8 -e log.data.grp8 ./get_v16.data.sh grp8)
        DEPEND="-d afterok:$DATAH:$DATA1:$DATA2:$DATA3:$DATA4:$DATA5:$DATA6:$DATA7:$DATA8"
        DEPEND="-d afterok:$DATAH:$DATA1:$DATA2"
      fi
      ;;
 esac

else  # do not extract data.

  DEPEND=' '

fi  # extract data?

### Backup to HPSS
cat > ${CURDIR}/job_hpss_${gfs_ver}.sh << EOF
#!/bin/bash
##!/bin/bash --login
##SBATCH -J hpss-${cdate}
##SBATCH -A ${PROJECT_CODE} 
##SBATCH -n 1
##SBATCH -t 24:00:00
##SBATCH -p service
##SBATCH -D ./
##SBATCH -o ./hpss-${cdate}-out.txt
##SBATCH -e ./hpss-${cdate}-out.txt


module load hpss
set -x

echo "${cdate}" >> ${HPSS_EXTRACT_LOG}

heracntldir=${EXTRACT_DIR}/gdas.${yy}${mm}${dd}/${hh}
heraenkfdir=${EXTRACT_DIR}/enkfgdas.${yy}${mm}${dd}/${hh}
hpssdir=${HPSSDIR}/${gfs_ver}/${yy}${mm}
hsi "mkdir -p \${hpssdir}"

if [ -f \${heracntldir}/gdas.t${hh}z.atmanl.nemsio ] && [ -f \${heracntldir}/gdas.t${hh}z.sfcanl.nemsio ]; then
    cd \${heracntldir}
    htar -cv -f \${hpssdir}/gdas.${cdate}.tar *
    err=\$?
    if [ \${err} != '0' ]; then
        echo "HTAR cntl failed at \${cdate} and exit."
        exit \${err}
    else
        echo "HTAR cntl succeeded at ${cdate}."
    fi
else
    echo "\${heracntldir}/gdas.t${hh}z.atmanl.nemsio does not exist and exit"
    exit 1
fi


nens1=\`ls \${heraenkfdir}/mem???/gdas.t${hh}z.ratmanl.nemsio | wc -l\`
nens2=\`ls \${heraenkfdir}/mem???/RESTART/*sfcanl_data.tile6.nc | wc -l\`
if [ \${nens1} -eq 80 ] && [ \${nens2} -eq 80 ]; then
    cd \${heraenkfdir}
    htar -cv -f \${hpssdir}/enkfgdas.${cdate}.tar *
    err=\$?
    if [ \${err} != '0' ]; then
        echo "HTAR enkf failed at \${cdate} and exit."
        exit \${err}
    else
        echo "HTAR enkf succeeded at ${cdate}."
        echo "${cdate}" >> ${HPSS_HTAR_LOG}
        cdatep6=\`${NDATE} 6 ${cdate}\`
        echo "\${cdatep6}" > ${CYCLE_LOG}
        rm -rf \${heracntldir}
        rm -rf \${heraenkfdir}
cd ${CURDIR}
${DRIVER}
        exit \${err}
    fi
else
    echo "\${heraenkfdir} is not complete and exit"
    exit 1
fi
EOF

sbatch --parsable --partition=service --ntasks=1  -t $WALLT -A $PROJECT_CODE -q $QUEUE -J ${gfs_ver}_${cdate} \
         -o ${EXTRACT_DIR}/../log.hpss.${cdate} -e ${EXTRACT_DIR}/../log.hpss.${cdate}  ${DEPEND} ${CURDIR}/job_hpss_${gfs_ver}.sh
err=$?
if [ ${err} != '0' ]; then
    echo "HPSS job submission failed at \${cdate} and exit."
    exit ${err}
fi

exit 0
##### Not used below #####

if [ $RUN_CHGRES == yes ]; then

  export APRUN=srun
  NODES=3
  WALLT="0:15:00"
  export OMP_NUM_THREADS=1
  if [ $CRES_HIRES == 'C768' ] ; then
    NODES=5
  elif [ $CRES_HIRES == 'C1152' ] ; then
    NODES=8
    WALLT="0:20:00"
  fi
  case $gfs_ver in
    v12 | v13)
      export OMP_NUM_THREADS=4
      export OMP_STACKSIZE=1024M
      sbatch --parsable --ntasks-per-node=6 --nodes=${NODES} --cpus-per-task=$OMP_NUM_THREADS \
        -t $WALLT -A $PROJECT_CODE -q $QUEUE -J chgres_${CDUMP} \
        -o log.${CDUMP} -e log.${CDUMP} ${DEPEND} run_pre-v14.chgres.sh ${CDUMP}
      ;;
    v14)
      sbatch --parsable --ntasks-per-node=6 --nodes=${NODES} -t $WALLT -A $PROJECT_CODE -q $QUEUE -J chgres_${CDUMP} \
      -o log.${CDUMP} -e log.${CDUMP} ${DEPEND} run_v14.chgres.sh ${CDUMP}
      ;;
    v15)
      if [ "$CDUMP" = "gdas" ]; then
        sbatch --parsable --ntasks-per-node=6 --nodes=${NODES} -t $WALLT -A $PROJECT_CODE -q $QUEUE -J chgres_${CDUMP} \
        -o log.${CDUMP} -e log.${CDUMP} ${DEPEND} run_v15.chgres.sh ${CDUMP}
      else
        sbatch --parsable --ntasks-per-node=6 --nodes=${NODES} -t $WALLT -A $PROJECT_CODE -q $QUEUE -J chgres_${CDUMP} \
        -o log.${CDUMP} -e log.${CDUMP} ${DEPEND} run_v15.chgres.gfs.sh
      fi
      ;;
    v16retro)
      if [ "$CDUMP" = "gdas" ] ; then
        sbatch --parsable --ntasks-per-node=6 --nodes=${NODES} -t $WALLT -A $PROJECT_CODE -q $QUEUE -J chgres_${CDUMP} \
        -o log.${CDUMP} -e log.${CDUMP} ${DEPEND} run_v16retro.chgres.sh hires
      else
        sbatch --parsable --ntasks-per-node=6 --nodes=${NODES} -t $WALLT -A $PROJECT_CODE -q $QUEUE -J chgres_${CDUMP} \
        -o log.${CDUMP} -e log.${CDUMP} ${DEPEND} run_v16.chgres.sh ${CDUMP}
      fi
      ;;
    v16)
      sbatch --parsable --ntasks-per-node=6 --nodes=${NODES} -t $WALLT -A $PROJECT_CODE -q $QUEUE -J chgres_${CDUMP} \
      -o log.${CDUMP} -e log.${CDUMP} ${DEPEND} run_v16.chgres.sh ${CDUMP}
      ;;
  esac

  if [ "$CDUMP" = "gdas" ]; then

    WALLT="0:15:00"

    if [ "$gfs_ver" = "v16retro" ]; then

      sbatch --parsable --ntasks-per-node=12 --nodes=1 -t $WALLT -A $PROJECT_CODE -q $QUEUE -J chgres_enkf \
      -o log.enkf -e log.enkf ${DEPEND} run_v16retro.chgres.sh enkf

    else

      MEMBER=1
      #while [ $MEMBER -le 80 ]; do
      while [ $MEMBER -le 20 ]; do
        if [ $MEMBER -lt 10 ]; then
          MEMBER_CH="00${MEMBER}"
        else
          MEMBER_CH="0${MEMBER}"
        fi
        case $gfs_ver in
          v12 | v13)
              export OMP_NUM_THREADS=2
              export OMP_STACKSIZE=1024M
              sbatch --parsable --ntasks-per-node=12 --nodes=1 --cpus-per-task=$OMP_NUM_THREADS \
               -t $WALLT -A $PROJECT_CODE -q $QUEUE -J chgres_${MEMBER_CH} \
               -o log.${MEMBER_CH} -e log.${MEMBER_CH} ${DEPEND} run_pre-v14.chgres.sh ${MEMBER_CH}
            ;;
          v14)
              sbatch --parsable --ntasks-per-node=12 --nodes=1 -t $WALLT -A $PROJECT_CODE -q $QUEUE -J chgres_${MEMBER_CH} \
              -o log.${MEMBER_CH} -e log.${MEMBER_CH} ${DEPEND} run_v14.chgres.sh ${MEMBER_CH}
            ;;
          v15)
              sbatch --parsable --ntasks-per-node=12 --nodes=1 -t $WALLT -A $PROJECT_CODE -q $QUEUE -J chgres_${MEMBER_CH} \
              -o log.${MEMBER_CH} -e log.${MEMBER_CH} ${DEPEND} run_v15.chgres.sh ${MEMBER_CH}
            ;;
          v16)
              sbatch --parsable --ntasks-per-node=12 --nodes=1 -t $WALLT -A $PROJECT_CODE -q $QUEUE -J chgres_${MEMBER_CH} \
              -o log.${MEMBER_CH} -e log.${MEMBER_CH} ${DEPEND} run_v16.chgres.sh ${MEMBER_CH}
            ;;
        esac
        MEMBER=$(( $MEMBER + 1 ))
      done

    fi # v16 retro?

  fi  # which CDUMP?

fi  # run chgres?


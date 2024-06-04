#!/bin/sh
#PBS -N jobhafsgraph
#PBS -A HAFS-DEV
#PBS -q dev
#PBS -l select=4:mpiprocs=60:ompthreads=1:ncpus=60:mem=500G
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o jobhafsgraph.log

set -xe

date

cd $PBS_O_WORKDIR

export TOTAL_TASKS=${TOTAL_TASKS:-${SLURM_NTASKS:-240}}
export NCTSK=${NCTSK:-60}
export NCNODE=${NCNODE:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

export MPLBACKEND=agg

plotATCF=${plotATCF:-yes}
plotAtmos=${plotAtmos:-yes}
plotOcean=${plotOcean:-yes}

YMDH=${YMDH:-2022092000}
STORM=${STORM:-FIONA}
STORMID=${STORMID:-07L}
stormModel=${stormModel:-HFSA}
fhhhAll=$(seq -f "f%03g" 0 3 126)

modelLabels="['BEST','OFCL','${stormModel}','HWRF','HMON','AVNO']"
modelColors="['black','red','cyan','purple','green','blue']"
modelMarkers="['hr','.','.','.','.','.']"
modelMarkerSizes="[18,15,15,15,15,15]"
nset=""

ymdh=${YMDH}
stormname=${STORM}
stormid=${STORMID}
STORMID=`echo ${stormid} | tr '[a-z]' '[A-Z]' `
stormid=`echo ${stormid} | tr '[A-Z]' '[a-z]' `
STORMNAME=`echo ${stormname} | tr '[a-z]' '[A-Z]' `
stormname=`echo ${stormname} | tr '[A-Z]' '[a-z]' `

stormnmid=`echo ${stormname}${stormid} | tr '[A-Z]' '[a-z]' `
STORMNMID=`echo ${stormnmid} | tr '[a-z]' '[A-Z]' `
STORMNM=${STORMNMID:0:-3}
stormnm=${STORMNM,,}
STID=${STORMNMID: -3}
stid=${STID,,}
STORMNUM=${STID:0:2}
BASIN1C=${STID: -1}
basin1c=${BASIN1C,,}
yyyy=`echo ${ymdh} | cut -c1-4`
ymd=`echo ${ymdh} | cut -c1-8`
hh=`echo ${ymdh} | cut -c9-10`

if [ ${stormModel} = "HFSA" ] && [ "lec" = "*${basin1c}*" ]; then
  plotWave=${plotWave:-yes}
else
  plotWave=${plotWave:-no}
fi

#HOMEgraph=/your/graph/home/dir
#WORKgraph=/your/graph/work/dir # if not specified, a default location relative to COMhafs will be used
#COMgraph=/your/graph/com/dir   # if not specified, a default location relative to COMhafs will be used
#COMhafs=/your/hafs/com/dir

export HOMEgraph=${HOMEgraph:-$(pwd)/../}
export USHgraph=${USHgraph:-${HOMEgraph}/ush}
export DRIVERATMOS=${USHgraph}/driverAtmos.sh
export DRIVEROCEAN=${USHgraph}/driverOcean.sh

export COMhafs=$(compath.py ${envir}/${NET}/${hafs_ver})/${RUN}.${ymd}/${hh}
export WORKgraph=${WORKgraph:-/lfs/h2/emc/ptmp/${USER}/workgraph_${RUN,,}}
export COMgraph=${COMgraph:-/lfs/h2/emc/ptmp/${USER}/comgraph_${RUN,,}}

atcfFile=${COMhafs}/${stormid}.${ymdh}.${stormModel,,}.trak.atcfunix

source ${USHgraph}/graph_pre_job.sh.inc
export machine=${WHERE_AM_I:-wcoss2} # platforms: wcoss2, hera, orion, jet
if [ ${machine} = jet ]; then
  export ADECKgraph=${ADECKgraph:-/mnt/lfs4/HFIP/hwrf-data/hwrf-input/abdeck/aid}
  export BDECKgraph=${BDECKgraph:-/mnt/lfs4/HFIP/hwrf-data/hwrf-input/abdeck/btk}
  export cartopyDataDir=${cartopyDataDir:-/mnt/lfs4/HFIP/hwrfv3/local/share/cartopy}
elif [ ${machine} = hera ]; then
  export ADECKgraph=${ADECKgraph:-/scratch1/NCEPDEV/hwrf/noscrub/input/abdeck/aid}
  export BDECKgraph=${BDECKgraph:-/scratch1/NCEPDEV/hwrf/noscrub/input/abdeck/btk}
  export cartopyDataDir=${cartopyDataDir:-/scratch1/NCEPDEV/hwrf/noscrub/local/share/cartopy}
elif [ ${machine} = orion ]; then
  export ADECKgraph=${ADECKgraph:-/work/noaa/hwrf/noscrub/input/abdeck/aid}
  export BDECKgraph=${BDECKgraph:-/work/noaa/hwrf/noscrub/input/abdeck/btk}
  export cartopyDataDir=${cartopyDataDir:-/work/noaa/hwrf/noscrub/local/share/cartopy}
elif [ ${machine} = wcoss2 ]; then
  export ADECKgraph=${ADECKgraph:-/lfs/h2/emc/hur/noscrub/input/abdeck/aid}
  export BDECKgraph=${BDECKgraph:-/lfs/h2/emc/hur/noscrub/input/abdeck/btk}
  export cartopyDataDir=${cartopyDataDir:-/lfs/h2/emc/hur/noscrub/local/share/cartopy}
else
  export ADECKgraph=${ADECKgraph:-/your/abdeck/aid}
  export BDECKgraph=${BDECKgraph:-/your/abdeck/btk}
  export cartopyDataDir=${cartopyDataDir:-/your/local/share/cartopy}
fi

source ${USHgraph}/graph_runcmd.sh.inc

mkdir -p ${WORKgraph}
cd ${WORKgraph}

if [ ${basin1c} = 'l' ]; then
  basin2c='al'
  BASIN2C='AL'
  BASIN='NATL'
elif [ ${basin1c} = 'e' ]; then
  basin2c='ep'
  BASIN2C='EP'
  BASIN='EPAC'
elif [ ${basin1c} = 'c' ]; then
  basin2c='cp'
  BASIN2C='CP'
  BASIN='CPAC'
elif [ ${basin1c} = 'w' ]; then
  basin2c='wp'
  BASIN2C='WP'
  BASIN='WPAC'
elif [ ${basin1c} = 's' ] || [ ${basin1c} = 'p'  ]; then
  basin2c='sh'
  BASIN2C='SH'
  BASIN='SH'
elif [ ${basin1c} = 'a' ] || [ ${basin1c} = 'b'  ]; then
  basin2c='io'
  BASIN2C='IO'
  BASIN='NIO'
else
  echo "WRONG BASIN DESIGNATION basin1c=${basin1c}"
  echo 'SCRIPT WILL EXIT'
  exit 1
fi

work_dir="${WORKgraph}"
archbase="${COMgraph}/figures"
archdir="${archbase}/RT${yyyy}_${BASIN}/${STORMNM}${STID}/${STORMNM}${STID}.${ymdh}"

mkdir -p ${work_dir}
cd ${work_dir}

#==============================================================================
# For the ATCF figures

if [ ${plotATCF} = yes ]; then

if [ -f ${atcfFile} ]; then
  atcfFile=${atcfFile}
elif [ -f ${atcfFile%.all} ]; then
  atcfFile=${atcfFile%.all}
else
  echo "File ${atcfFile} does not exist"
  echo 'SCRIPT WILL EXIT'
  exit 1
fi

# make the track and intensity plots
sh ${HOMEgraph}/ush/python/ATCF/plotATCF.sh ${STORMNM} ${STID} ${ymdh} ${stormModel} ${COMhafs} ${ADECKgraph} ${BDECKgraph} ${HOMEgraph}/ush/python ${WORKgraph} ${archdir} ${modelLabels} ${modelColors} ${modelMarkers} ${modelMarkerSizes} ${nset}

date

fi
#==============================================================================
# For the atmos figures

if [ ${plotAtmos} = yes ]; then

#Generate the cmdfile
cmdfile="cmdfile.$STORM$STORMID.$YMDH"
rm -f $cmdfile
touch $cmdfile

for fhhh in ${fhhhAll}; do

for stormDomain in parent storm; do

if [ ${stormDomain} = "parent" ]; then
  figScriptAll=( \
    plot_mslp_wind10m.py \
    plot_tsfc_mslp_wind10m.py \
    plot_t2m_mslp_wind10m.py \
    plot_heatflux_wind10m.py \
    plot_shtflux_wind10m.py \
    plot_lhtflux_wind10m.py \
    plot_precip_mslp_thk.py \
    plot_reflectivity.py \
    plot_goes_ir13.py \
    plot_goes_wv9.py \
    plot_ssmisf17_mw37ghz.py \
    plot_ssmisf17_mw91ghz.py \
    plot_850mb_200mb_vws.py \
    plot_rhmidlev_hgt_wind.py \
    plot_temp_hgt_wind.py \
    plot_temp_hgt_wind.py \
    plot_temp_hgt_wind.py \
    plot_temp_hgt_wind.py \
    plot_rh_hgt_wind.py \
    plot_rh_hgt_wind.py \
    plot_rh_hgt_wind.py \
    plot_rh_hgt_wind.py \
    plot_vort_hgt_wind.py \
    plot_vort_hgt_wind.py \
    plot_vort_hgt_wind.py \
    plot_vort_hgt_wind.py \
    plot_streamline_wind.py \
    )
  levAll=( \
    1003 \
    1003 \
    1003 \
    1003 \
    1003 \
    1003 \
    1003 \
    1003 \
    1003 \
    1003 \
    1003 \
    1003 \
    1003 \
    1003 \
    850 \
    700 \
    500 \
    200 \
    850 \
    700 \
    500 \
    200 \
    850 \
    700 \
    500 \
    200 \
    850 \
    1003 \
    1003 \
    1003 \
    )
elif [ ${stormDomain} = "storm" ]; then
  figScriptAll=( \
    plot_mslp_wind10m.py \
    plot_tsfc_mslp_wind10m.py \
    plot_t2m_mslp_wind10m.py \
    plot_heatflux_wind10m.py \
    plot_shtflux_wind10m.py \
    plot_lhtflux_wind10m.py \
    plot_precip_mslp_thk.py \
    plot_reflectivity.py \
    plot_goes_ir13.py \
    plot_goes_wv9.py \
    plot_ssmisf17_mw37ghz.py \
    plot_ssmisf17_mw91ghz.py \
    plot_rhmidlev_hgt_wind.py \
    plot_temp_hgt_wind.py \
    plot_temp_hgt_wind.py \
    plot_temp_hgt_wind.py \
    plot_temp_hgt_wind.py \
    plot_rh_hgt_wind.py \
    plot_rh_hgt_wind.py \
    plot_rh_hgt_wind.py \
    plot_rh_hgt_wind.py \
    plot_vort_hgt_wind.py \
    plot_vort_hgt_wind.py \
    plot_vort_hgt_wind.py \
    plot_vort_hgt_wind.py \
    plot_streamline_wind.py \
    plot_tempanomaly_hgt_wind.py \
	plot_crs_sn_wind.py \
    plot_crs_sn_rh_tempanomaly.py \
    plot_crs_sn_reflectivity.py \
	plot_crs_we_wind.py \
    plot_crs_we_rh_tempanomaly.py \
    plot_crs_we_reflectivity.py \
    plot_azimuth_wind.py \
    plot_azimuth_tempanomaly.py \
    plot_azimuth_rh_q.py \
    plot_azimuth_reflectivity.py \
    )
  levAll=( \
    1003 \
    1003 \
    1003 \
    1003 \
    1003 \
    1003 \
    1003 \
    1003 \
    1003 \
    1003 \
    1003 \
    1003 \
    1003 \
    850 \
    700 \
    500 \
    200 \
    850 \
    700 \
    500 \
    200 \
    850 \
    700 \
    500 \
    200 \
    850 \
    200 \
    1003 \
    1003 \
    1003 \
    1003 \
    1003 \
    1003 \
    1003 \
    1003 \
    1003 \
    1003 \
    )
fi
nscripts=${#figScriptAll[*]}

for((i=0;i<${nscripts};i++)); do
  echo ${figScriptAll[$i]} ${levAll[$i]}
# echo "${APRUNS} ${DRIVERATMOS} $stormModel $STORM $STORMID $YMDH $stormDomain ${figScriptAll[$i]} ${levAll[$i]} $fhhh > ${WORKgraph}/$STORM$STORMID.$YMDH.${stormDomain}.${figScriptAll[$i]%.*}.${fhhh}.log 2>&1 ${BACKGROUND}" >> $cmdfile
  echo "time ${DRIVERATMOS} $stormModel $STORM $STORMID $YMDH $stormDomain ${figScriptAll[$i]} ${levAll[$i]} $fhhh > ${WORKgraph}/$STORM$STORMID.$YMDH.${stormDomain}.${figScriptAll[$i]%.*}.${fhhh}.log 2>&1" >> $cmdfile
done

done

done

# For the atm swath figures
fhhh=f126
stormDomain=parent
figScriptAll=( \
  plot_precip_swath.py \
  plot_wind_swath.py \
  )
levAll=( \
  1003 \
  1003 \
  )
nscripts=${#figScriptAll[*]}

for((i=0;i<${nscripts};i++)); do
  echo ${figScriptAll[$i]} ${levAll[$i]}
  echo "time ${DRIVERATMOS} $stormModel $STORM $STORMID $YMDH $stormDomain ${figScriptAll[$i]} ${levAll[$i]} $fhhh > ${WORKgraph}/$STORM$STORMID.$YMDH.${stormDomain}.${figScriptAll[$i]%.*}.${fhhh}.log 2>&1" >> $cmdfile
done

chmod u+x ./$cmdfile
if [ ${machine} = "wcoss2" ]; then
  ncmd=$(cat ./$cmdfile | wc -l)
  ncmd_max=$((ncmd < TOTAL_TASKS ? ncmd : TOTAL_TASKS))
  $APRUNCFP -n $ncmd_max cfp ./$cmdfile
else
  ${APRUNC} ${MPISERIAL} -m ./$cmdfile
fi

date

fi # if [ ${plotAtmos} = yes ]; then

#==============================================================================
# For the wave figures

if [ ${plotWave} = yes ]; then

#Generate the cmdfile
cmdfile="cmdfile_wave.$STORM$STORMID.$YMDH"
rm -f $cmdfile
touch $cmdfile

for fhhh in ${fhhhAll}; do

for stormDomain in parent storm; do

if [ ${stormDomain} = "parent" ]; then
  figScriptAll=( \
    plot_wave_hs.py \
    plot_wave_tm.py \
    plot_wave_tp.py \
    )
  levAll=( \
    1003 \
    1003 \
    1003 \
    )
elif [ ${stormDomain} = "storm" ]; then
  figScriptAll=( \
    plot_wave_hs.py \
    plot_wave_tm.py \
    plot_wave_tp.py \
    )
  levAll=( \
    1003 \
    1003 \
    1003 \
    )
fi
nscripts=${#figScriptAll[*]}

for((i=0;i<${nscripts};i++)); do
  echo ${figScriptAll[$i]} ${levAll[$i]}
  echo "time ${DRIVERATMOS} $stormModel $STORM $STORMID $YMDH $stormDomain ${figScriptAll[$i]} ${levAll[$i]} $fhhh > ${WORKgraph}/$STORM$STORMID.$YMDH.${stormDomain}.${figScriptAll[$i]%.*}.${fhhh}.log 2>&1" >> $cmdfile
done

done

done

chmod u+x ./$cmdfile
if [ ${machine} = "wcoss2" ]; then
  ncmd=$(cat ./$cmdfile | wc -l)
  ncmd_max=$((ncmd < TOTAL_TASKS ? ncmd : TOTAL_TASKS))
  $APRUNCFP -n $ncmd_max cfp ./$cmdfile
else
  ${APRUNC} ${MPISERIAL} -m ./$cmdfile
fi

date

fi # if [ ${plotWave} = yes ]; then

#==============================================================================
# For the ocean figures

if [ ${plotOcean} = 'yes' ]; then

is6Hr=${is6Hr:-False}
trackOn=${trackOn:-True}
figTimeLevels=$(seq 0 42)
#is6Hr=${is6Hr:-True}
#figTimeLevels=$(seq 0 20)

#Generate the cmdfile
cmdfile="cmdfile_ocean.$STORM$STORMID.$YMDH"
rm -f $cmdfile
touch $cmdfile

figScriptAll=( \
  "plot_sst.py" \
  "plot_sss.py" \
  "plot_mld.py" \
  "plot_ohc.py" \
  "plot_z20.py" \
  "plot_z26.py" \
  "plot_storm_sst.py" \
  "plot_storm_sss.py" \
  "plot_storm_mld.py" \
  "plot_storm_ohc.py" \
  "plot_storm_z20.py" \
  "plot_storm_z26.py" \
  "plot_storm_tempz40m.py" \
  "plot_storm_tempz70m.py" \
  "plot_storm_tempz100m.py" \
  "plot_storm_wvelz40m.py" \
  "plot_storm_wvelz70m.py" \
  "plot_storm_wvelz100m.py" \
  )

nscripts=${#figScriptAll[*]}

for((i=0;i<${nscripts};i++));
do

  echo ${figScriptAll[$i]}
# echo "${APRUNS} ${DRIVEROCEAN} $stormModel $STORM $STORMID $YMDH $trackOn ${figScriptAll[$i]} > ${WORKgraph}/$STORM$STORMID.$YMDH.${figScriptAll[$i]%.*}.log 2>&1 ${BACKGROUND}" >> $cmdfile
  echo "time ${DRIVEROCEAN} $stormModel $STORM $STORMID $YMDH $trackOn ${figScriptAll[$i]} > ${WORKgraph}/$STORM$STORMID.$YMDH.${figScriptAll[$i]%.*}.log 2>&1" >> $cmdfile

done

chmod u+x ./$cmdfile
if [ ${machine} = "wcoss2" ]; then
  ncmd=$(cat ./$cmdfile | wc -l)
  ncmd_max=$((ncmd < TOTAL_TASKS ? ncmd : TOTAL_TASKS))
  $APRUNCFP -n $ncmd_max cfp ./$cmdfile
else
  ${APRUNC} ${MPISERIAL} ./$cmdfile
fi

date

fi # if [ ${plotOcean} = yes ]; then

#==============================================================================

echo 'job done'

#!/bin/sh
##
## SCRIPT FOR SINGLE DAY PROCESSING AND MAP GENERATION
##
## YYYYMMDD is the start day of NAQFC PM   simulation
## Cycle_hr is the model run starting hour
##
module use /apps/test/lmodules/core/
module load GrADS/2.2.2
module load prod_util
module load prod_envir
module load grib_util
wgrib=/apps/ops/prod/libs/intel/19.1.3.304/grib_util/1.2.2/bin/wgrib
wgrib2=/apps/ops/prod/libs/intel/19.1.3.304/wgrib2/2.0.8/bin/wgrib2
export GAUDPT=/apps/test/grads/spack/opt/spack/cray-sles15-zen2/gcc-11.2.0/grads-2.2.2-wckmyzg7qh5smosf6och6ehqtqlxoy4f/data/udpt
export GADDIR=/apps/test/grads/spack/opt/spack/cray-sles15-zen2/gcc-11.2.0/grads-2.2.2-wckmyzg7qh5smosf6och6ehqtqlxoy4f/data
## set -x
hl=`hostname | cut -c1`
if [ "${hl}" == "v" ]; then
  phase12_id='g'
else
  phase12_id='t'
fi

flag_test=yes
flag_test=no

flag_qsub=no
flag_qsub=yes

if [ "${flag_qsub}" == "yes" ]; then
   flag_scp=no
else
   flag_scp=yes
fi

TODAY=`date +%Y%m%d`

MSG="USAGE : $0 NCO_run (default:prod|para) Cycle_hr (default:all|06|12) YYYYMMDD_BEG YYYYMMDD_END"

exp=prod
sel_cyc=all
FIRSTDAY=${TODAY}
LASTDAY=${TODAY}

if [ $# -lt 2 ]; then
   echo $MSG
   exit
else
   exp=$1
   sel_cyc=$2
fi
if [ $# -lt 3 ]; then
   FIRSTDAY=${TODAY}
   LASTDAY=${TODAY}
elif [ $# -lt 4 ]; then
   FIRSTDAY=$3
   LASTDAY=$3
else
   FIRSTDAY=$3
   LASTDAY=$4
fi
case ${sel_cyc} in
   ''|*[!0-9]*) if [ "${sel_cyc}" == "all" ]; then
            declare -a cyc_opt=( 06 12 )
         else
            echo "input choice for cycle time is not defined $2, program stop"
            echo $MSG
            exit
         fi ;;
   *) cyc_in=`printf %2.2d $2`
      if [ "${cyc_in}" == "06" ] || [ "${cyc_in}" == "12" ]; then
         declare -a cyc_opt=( ${cyc_in} )
      else
         echo "input choice for cycle time is not defined $2, program stop"
         echo $MSG
         exit
      fi ;;
esac
echo ${cyc_opt[@]}

capexp=`echo ${exp} | tr '[:lower:]' '[:upper:]'`
if [ ${exp} == 'para1' ]; then flag_update=no; fi

## echo " ${exp} ${sel_cyc} ${FIRSTDAY} ${LASTDAY}"

remote_dir=/home/people/emc/www/htdocs/mmb/hchuang/web/fig
remote_host=emcrzdm.ncep.noaa.gov
remote_user=hchuang

grid148=148
grid227=227

flag_update=no
if [ "${LASTDAY}" == "${TODAY}" ]; then flag_update=yes; fi

gs_dir=/lfs/h2/emc/physics/noscrub/${USER}/plot/cmaq
gs_dir=`pwd`

declare -a reg=( dset conus east west  ne10  nw10  se10  swse  ak   hi   )
declare -a lon0=( -175 -133 -100 -130  -81   -125  -91   -125  -170 -161 )
declare -a lon1=(   55  -60  -60  -90  -66   -105  -74   -100  -130 -154 )
declare -a lat0=(    0   21   24   21   37     37   24     21    50   18 )
declare -a lat1=(   80   52   50   50   48     50   40     45    80   23 )
nreg=${#reg[@]}
let max_ireg=${nreg}-1
idset=0
iconus=1
ieast=2
iwest=3
ine10=4
inw10=5
ise10=6
iswse=7
iak=8
ihi=9

declare -a mchr=( JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC )

declare -a typ=( aot )
ntyp=${#typ[@]}

capexp=`echo ${exp} | tr '[:lower:]' '[:upper:]'`
if [ ${exp} == 'para1' ]; then flag_update=no; fi

plotexp=${exp}
plotcapexp=${capexp}
mfileid="aot"
hfileid=${mfileid}
if [ $# -gt 4 ]; then
   flag_bicor=$5
   if [ "${flag_bicor}" == "bc" ]; then
      plotexp=${exp}"bc"
      plotcapexp=${capexp}"_BC"
      mfileid="aot_bc"
      hfileid=${mfileid}
   fi
fi
   
NOW=${FIRSTDAY}
while [ ${NOW} -le ${LASTDAY} ]; do

   if [ ${exp} == 'prod' ]; then
      comdir=/com2/aqm/${exp}/cs.${NOW}
      comdir2=/lfs/h2/emc/physics/noscrub/${USER}/com/aqm/${exp}/cs.${NOW}
      comdir2=/lfs/h2/emc/physics/noscrub/${USER}/com/aqm/${exp}/cs.${NOW}
      comdir=/lfs/h1/ops/${exp}/com/aqm/v6.1/cs.${NOW}
      comdir2=/lfs/h2/emc/physics/noscrub/${USER}/com/aqm/${exp}/cs.${NOW}
      if [ ! -d ${comdir} ]; then
         if [ -d ${comdir2} ]; then
            comdir=${comdir2}
         else
            echo "Can not find ${comdir} or ${comdir2}, program stop"
            exit
         fi
      fi
   elif [ ${exp} == 'para' ]; then
      comdir=/com2/aqm/${exp}/cs.${NOW}
      comdir=/lfs/h1/ops/${exp}/com/aqm/v6.1/cs.${NOW}
      comdir=/lfs/h2/emc/ptmp/${USER}/com/aqm/${exp}/cs.${NOW}
      comdir=/lfs/h2/emc/physics/noscrub/${USER}/com/aqm/${exp}/cs.${NOW}
   else
      comdir=/lfs/h2/emc/physics/noscrub/${USER}/com/aqm/${exp}/cs.${NOW}
   fi
   if [ ! -d ${comdir} ]; then
      echo "////////////////////////////////////////"
      echo "${comdir} does not existed, program stop"
      echo "////////////////////////////////////////"
      exit
   else
      echo "////////////////////////////////////////////"
      echo "/// Fetching PM data Data from ${comdir} ///"
      echo "////////////////////////////////////////////"
   fi
   for cych in "${cyc_opt[@]}"; do
      cdate=${NOW}${cych}
      F1=$(${NDATE} +1 ${cdate}| cut -c9-10)
      TOMORROW=$(${NDATE} +24 ${cdate}| cut -c1-8)
      THIRDDAY=$(${NDATE} +48 ${cdate}| cut -c1-8)
      FOURTHDAY=$(${NDATE} +72 ${cdate}| cut -c1-8)
   
      Y1=`echo ${NOW} | cut -c1-4`
      MX=`echo ${NOW} | cut -c5-5`
      if [ ${MX} == '0' ]; then
         M1=`echo ${NOW} | cut -c6-6`
      else
         M1=`echo ${NOW} | cut -c5-6`
      fi
      D1=`echo ${NOW} | cut -c7-8`
      Y2=`echo ${TOMORROW} | cut -c1-4`
      MX=`echo ${TOMORROW} | cut -c5-5`
      if [ ${MX} == '0' ]; then
         M2=`echo ${TOMORROW} | cut -c6-6`
      else
         M2=`echo ${TOMORROW} | cut -c5-6`
      fi
      D2=`echo ${TOMORROW} | cut -c7-8`
      Y3=`echo ${THIRDDAY} | cut -c1-4`
      MX=`echo ${THIRDDAY} | cut -c5-5`
      if [ ${MX} == '0' ]; then
         M3=`echo ${THIRDDAY} | cut -c6-6`
      else
         M3=`echo ${THIRDDAY} | cut -c5-6`
      fi
      D3=`echo ${THIRDDAY} | cut -c7-8`
      Y4=`echo ${FOURTHDAY} | cut -c1-4`
      MX=`echo ${FOURTHDAY} | cut -c5-5`
      if [ ${MX} == '0' ]; then
         M4=`echo ${FOURTHDAY} | cut -c6-6`
      else
         M4=`echo ${FOURTHDAY} | cut -c5-6`
      fi
      D4=`echo ${FOURTHDAY} | cut -c7-8`
      range1=05Z${D1}${mchr[$M1-1]}${Y1}-04Z${D2}${mchr[$M2-1]}${Y2}
      range2=05Z${D2}${mchr[$M2-1]}${Y2}-04Z${D3}${mchr[$M3-1]}${Y3}
      range3=05Z${D3}${mchr[$M3-1]}${Y3}-04Z${D4}${mchr[$M4-1]}${Y4}

      tmpdir=/lfs/h2/emc/stmp/${USER}/com2_aqm_${plotexp}_pm_max.${NOW}${cych}
      if [ -d ${tmpdir} ]; then
         /bin/rm -f ${tmpdir}/*
      else
         mkdir -p ${tmpdir}
      fi
   
      fig_dir=/lfs/h2/emc/stmp/${USER}/daily_plot_${mfileid}/aqm_${plotexp}_pm
      outdir=${fig_dir}.${NOW}${cych}
      if [ ! -d ${outdir} ]; then mkdir -p ${outdir}; fi

      let numcyc=${cych}
      cychr="t${cych}z"
      echo " Perform operation on cych = ${cych}  cychr = ${cychr}"
      if [ "${flag_test}" == "yes" ]; then continue; fi
      for i in "${typ[@]}"
      do
         case ${i} in
           aot)          cp ${comdir}/aqm.${cychr}.${mfileid}.f*.${grid148}.grib2      ${tmpdir};;
         esac
      done
      cd ${tmpdir}
   
      n0=0
      let n1=${ntyp}-1
      let ptyp=n0
      while [ ${ptyp} -le ${n1} ]; do

         if [ -e aqm.ctl ]; then /bin/rm -f aqm.ctl; fi
#
         if [ ${typ[${ptyp}]} = 'aot' ]; then 
   
            t0=1
            t1=48
            let numi=t0
            while [ ${numi} -le ${t1} ]; do
   
               fcsti=`printf %3.3d ${numi}`
               i=`printf %2.2d ${numi}`
   
               let j=numi+${numcyc}
               if [ ${j} -ge 72 ]; then
                  let j=j-72
                  date=${FOURTHDAY}
               elif [ ${j} -ge 48 ]; then
                  let j=j-48
                  date=${THIRDDAY}
               elif [ ${j} -ge 24 ]; then
                  let j=j-24
                  date=${TOMORROW}
               else
                  date=${NOW}
               fi
               numj=`printf %2.2d ${j}`
      
               YH=`echo ${date} | cut -c1-4`
               MX=`echo ${date} | cut -c5-5`
               if [ ${MX} == '0' ]; then
                  MH=`echo ${date} | cut -c6-6`
               else
                  MH=`echo ${date} | cut -c5-6`
               fi
               DH=`echo ${date} | cut -c7-8`
   

               Newdate=$(${NDATE} ${numi} ${NOW}${cych})
               Ynew=`echo ${Newdate} | cut -c1-4`
               Xnew=`echo ${Newdate} | cut -c5-5`
               if [ ${Xnew} == '0' ]; then
                  Mnew=`echo ${Newdate} | cut -c6-6`
               else
                  Mnew=`echo ${Newdate} | cut -c5-6`
               fi
               Dnew=`echo ${Newdate} | cut -c7-8`
               hnew=`echo ${Newdate} | cut -c9-10`
               if [ -e aqm.ctl ]; then /bin/rm -f aqm.ctl; fi
   
               cat >  aqm.ctl <<EOF
dset ${tmpdir}/aqm.${cychr}.${hfileid}.f${i}.${grid148}.grib2
index ${tmpdir}/aqm.${cychr}.${hfileid}.f${i}.${grid148}.grib2.idx
undef 9.999E+20
title aqm.${cychr}.${hfileid}.f${i}.${grid148}.grib2
* produced by alt_g2ctl v1.0.5, use alt_gmp to make idx file
* command line options: aqm.${cychr}.${hfileid}.f${i}.${grid148}.grib2 -short
* alt_gmp options: update=0
* alt_gmp options: nthreads=1
* alt_gmp options: big=0
* wgrib2 inventory flags: -npts -set_ext_name 1 -end_FT -ext_name -lev
* wgrib2 inv suffix: .invd01
* griddef=1:0:(442 x 265):grid_template=30:winds(grid): Lambert Conformal: (442 x 265) input WE:SN output WE:SN res 8 Lat1 21.821000 Lon1 239.372000 LoV 263.000000 LatD 33.000000 Latin1 33.000000 Latin2 45.000000 LatSP 0.000000 LonSP 0.000000 North P
dtype grib2
pdef 442 265 lccr 21.821000 -120.628 1 1 33.000000 45.000000 -97 12000.000000 12000.000000
xdef 620 linear -131.659038 0.117510583392794
ydef 293 linear 21.153709 0.109090909090909
tdef 1 linear ${hnew}Z${Dnew}${mchr[$Mnew-1]}${Ynew} 1mo
zdef 1 levels 1
vars 1
v1 0 0 "AOTK.aerosol=Ozone.aerosol_size_<0:1 sigma level"
ENDVARS
EOF
               alt_gmp aqm.ctl
   
               if [ -e aqm.plots ]; then /bin/rm -f aqm.plots; fi
   
               cat >  aqm.plots <<EOF
'reinit'
'set gxout shaded'
'set gxout grfill'
'set display color white'
'set mpdset hires'
'set grads off'
'set rgb 98 255 105 180'
'set rgb 79 240 240 240'
'set rgb 17  55  55 255'
'set rgb 18 110 110 255'
'set rgb 19 165 165 255'
'set rgb 20 220 220 255'
'c'
'open aqm.ctl'
'set lat ${lat0[${iconus}]} ${lat1[${iconus}]}'
'set lon ${lon0[${iconus}]} ${lon1[${iconus}]}'
'${gs_dir}/draw.aqm.aot.gs ${gs_dir} ${outdir} ${plotexp} ${plotcapexp} ${NOW} ${cychr} ${i} ${date}/${numj}00V${fcsti} conus'
'set lat ${lat0[${ieast}]} ${lat1[${ieast}]}'
'set lon ${lon0[${ieast}]} ${lon1[${ieast}]}'
'${gs_dir}/draw.aqm.aot.gs ${gs_dir} ${outdir} ${plotexp} ${plotcapexp} ${NOW} ${cychr} ${i} ${date}/${numj}00V${fcsti} east'
'set lat ${lat0[${iwest}]} ${lat1[${iwest}]}'
'set lon ${lon0[${iwest}]} ${lon1[${iwest}]}'
'${gs_dir}/draw.aqm.aot.gs ${gs_dir} ${outdir} ${plotexp} ${plotcapexp} ${NOW} ${cychr} ${i} ${date}/${numj}00V${fcsti} west'
'set lat ${lat0[${ine10}]} ${lat1[${ine10}]}'
'set lon ${lon0[${ine10}]} ${lon1[${ine10}]}'
'${gs_dir}/draw.aqm.aot.gs ${gs_dir} ${outdir} ${plotexp} ${plotcapexp} ${NOW} ${cychr} ${i} ${date}/${numj}00V${fcsti} ne'
'set lat ${lat0[${inw10}]} ${lat1[${inw10}]}'
'set lon ${lon0[${inw10}]} ${lon1[${inw10}]}'
'${gs_dir}/draw.aqm.aot.gs ${gs_dir} ${outdir} ${plotexp} ${plotcapexp} ${NOW} ${cychr} ${i} ${date}/${numj}00V${fcsti} nw'
'set lat ${lat0[${ise10}]} ${lat1[${ise10}]}'
'set lon ${lon0[${ise10}]} ${lon1[${ise10}]}'
'${gs_dir}/draw.aqm.aot.gs ${gs_dir} ${outdir} ${plotexp} ${plotcapexp} ${NOW} ${cychr} ${i} ${date}/${numj}00V${fcsti} se'
'set lat ${lat0[${iswse}]} ${lat1[${iswse}]}'
'set lon ${lon0[${iswse}]} ${lon1[${iswse}]}'
'${gs_dir}/draw.aqm.aot.gs ${gs_dir} ${outdir} ${plotexp} ${plotcapexp} ${NOW} ${cychr} ${i} ${date}/${numj}00V${fcsti} sw'
'quit'
EOF
               grads -blc "run aqm.plots"
               ((numi++))
            done
         fi
         ((ptyp++))
      done
      if [ ${flag_scp} == 'yes' ]; then  # for RZDM maintainence
         ##
         ## TRANSFER PLOTS TO RZDM
         ##
         scp ${outdir}/aqm*.png ${remote_user}@${remote_host}:${remote_dir}/${Y1}/${NOW}/${cychr}
      fi
   done   ## end for loop cych in "${cyc_opt[@]}"
   cdate=${NOW}"00"
   NOW=$(${NDATE} +24 ${cdate}| cut -c1-8)
done

if [ "${flag_qsub}" == "yes" ]; then
   working_dir=/lfs/h2/emc/stmp/${USER}/job_submit
   mkdir -p ${working_dir}
   cd ${working_dir}

   task_cpu='05:00:00'
   job_name=cmaq_${mfileid}max_${exp}${sel_cyc}
   batch_script=trans_cmaq${mfileid}max_${exp}.${FIRSTDAY}.${LASTDAY}.sh
   if [ -e ${batch_script} ]; then /bin/rm -f ${batch_script}; fi

   logdir=/lfs/h2/emc/ptmp/${USER}/batch_logs
   if [ ! -d ${logdir} ]; then mkdir -p ${logdir}; fi

   logfile=${logdir}/${job_name}_${FIRSTDAY}_${LASTDAY}.out
   if [ -e ${logfile} ]; then /bin/rm -f ${logfile}; fi

   file_hd=aqm
   file_type=png
   cat > ${batch_script} << EOF
#!/bin/sh
#PBS -o ${logfile}
#PBS -e ${logfile}
#PBS -l place=shared,select=1:ncpus=1:mem=4GB
#PBS -N j${job_name}
#PBS -q dev_transfer
#PBS -A AQM-DEV
#PBS -l walltime=${task_cpu}
# 
# 
#### 
##
##  Provide fix date daily Hysplit data processing
##
   module load prod_util/1.1.3

   FIRSTDAY=${FIRSTDAY}
   LASTDAY=${LASTDAY}
   exp=${exp}
   remote_user=${remote_user}
   remote_host=${remote_host}
   remote_dir=${remote_dir}
   file_hd=${file_hd}
   file_type=${file_type}
   flag_update=${flag_update}
   fig_dir=${fig_dir}
   declare -a cyc=( ${cyc_opt[@]} )
EOF
   ##
   ##  Creat part 2 script : exact wording of scripts
   ##
   cat > ${batch_script}.add  << 'EOF'

   NOW=${FIRSTDAY}
   while [ ${NOW} -le ${LASTDAY} ]; do
      YY=`echo ${NOW} | cut -c1-4`
      YM=`echo ${NOW} | cut -c1-6`

      for i in "${cyc[@]}"; do
         cycle=t${i}z
         data_dir=${fig_dir}.${NOW}${i}
         if [ -d ${data_dir} ]; then
            scp ${data_dir}/${file_hd}*${file_type} ${remote_user}@${remote_host}:${remote_dir}/${YY}/${NOW}/${cycle}
         else
            echo "Can not find ${data_dir}, skip to next cycle/day"
         fi
      done  ## cycle time
      cdate=${NOW}"00"
      NOW=$(${NDATE} +24 ${cdate}| cut -c1-8)
   done
   if [ "${flag_update}" == "yes" ]; then
      script_dir=/naqfc/save/${USER}/WEB/base
      cd ${script_dir}

      script_name=wcoss.run.cmaq_pm.sh
      bash ${script_name} ${LASTDAY}

      script_name=wcoss.run.cmaq2_pm.sh
      bash ${script_name} ${LASTDAY}
   fi
exit
EOF
   ##
   ##  Combine both working script together
   ##
   cat ${batch_script}.add >> ${batch_script}
   ##
   ##  Submit run scripts
   ##
   if [ "${flag_test}" == "no" ]; then
      qsub < ${batch_script}
   else
      echo "test qsub < ${batch_script} completed"
   fi
fi
exit

#!/bin/sh
#***********************************************************************
#*                   GNU Lesser General Public License
#*
#* This file is part of the SHiELD Build System.
#*
#* The SHiELD Build System free software: you can redistribute it
#* and/or modify it under the terms of the
#* GNU Lesser General Public License as published by the
#* Free Software Foundation, either version 3 of the License, or
#* (at your option) any later version.
#*
#* The SHiELD Build System distributed in the hope that it will be
#* useful, but WITHOUT ANYWARRANTY; without even the implied warranty
#* of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#* See the GNU General Public License for more details.
#*
#* You should have received a copy of the GNU Lesser General Public
#* License along with theSHiELD Build System
#* If not, see <http://www.gnu.org/licenses/>.
#***********************************************************************
#
#  DISCLAIMER: This script is provided as-is for the purpose of continuous
#              integration software regression testing and as such is
#              unsupported.
#
#SBATCH --ntasks=24

#Dry baroclinic wave in hydrostatic dynamics
#Also demonstrates the plev diagnostic capability

#export BUILDDIR=/home/links/ntl203/fv3_solo/SHiELD_build
export COMPILER=gnu
#export SCRATCHDIR=/scratch/ntl203/fv3_solo_work
export LAUNCHER=mpirun

RUN_DIR=`pwd`
SCRIPT="${RUN_DIR}/$0"

if [ -z ${FV3_BUILDDIR} ] ; then
  echo -e "\nERROR:\tset FV3_BUILDDIR to the base path /<path>/SHiELD_build/ \n"
  exit 99
fi

# when we are running this test for CI, we run it in a container
# an expected value for container would be "--mpi=pmi2 singularity exec -B <file directory> <container>"
# this is needed within the run command
if [ -z "$1" ] ; then
  CONTAINER=""
else
  CONTAINER=$1
  echo -e "\nThis test will be run inside of a container with the command $1"
fi

# configure the site
COMPILER=${COMPILER:-intel}
. ${FV3_BUILDDIR}/site/environment.${COMPILER}.sh

#set -x

# necessary for OpenMP
export OMP_STACKSIZE=256m

# case specific details
res=96
MEMO="solo.HS_new" # trying repro executable
TYPE="hydro"      # choices:  nh, hydro
MODE="32bit"      # choices:  32bit, 64bit
GRID="C$res"
HYPT="off"         # choices:  on, off  (controls hyperthreading)
COMP="prod"       # choices:  debug, repro, prod
NUM_TOT=6

# directory structure
 WORKDIR=${FV3_SCRATCHDIR:-${FV3_BUILDDIR}}/${GRID}.${MEMO}/
 executable=${FV3_BUILDDIR}/Build/bin/SOLO_${TYPE}.${COMP}.${MODE}.${COMPILER}.x

# changeable parameters
    # dycore definitions
    npx="97"
    npy="97"
    npz="32"
    layout_x="1"
    layout_y="3"
    io_layout="1,1" #Want to increase this in a production run??
    nthreads="2"

    # run length
    days="5"
    hours="0"
    minutes="0"
    seconds="0"
    dt_atmos="1800"

mkdir -p $WORKDIR/restart 
RST_COUNT=$WORKDIR/restart/rst.count 
if [ -f ${RST_COUNT} ] ; then 
    source ${RST_COUNT}
    RESTART_RUN="T"
else
    num=0
    RESTART_RUN="F"
fi 
echo 'I AM HERE' $num $RESTART_RUN
if [ ${RESTART_RUN} == "F" ] ; then 

    rm -rf $WORKDIR/rundir 
    
    mkdir -p $WORKDIR/rundir 
    cd $WORKDIR/rundir 
    
    mkdir -p RESTART 
    mkdir -p INPUT 

    # variables in input.nml for initial run
    warm_start=".F."
    mountain=".F."
    na_init=1
else 

    cd $WORKDIR/rundir 
    rm -rf INPUT/*
    
    #ln -s ${restart_dir}/*.res ${restart_dir}/[^0-9]*.nc INPUT/. # UNCOMMENT ME IF DOING SYMBOLIC LINK 
    mv RESTART/* INPUT/.
   
    warm_start=".T."
    mountain=".T."
    na_init=1
fi 
echo 'I AM HERE 2' $warm_start $mountain $na_init

ls INPUT/ 
ls RESTART/ 

if [ ${TYPE} = "nh" ] ; then
  # non-hydrostatic options
  make_nh=".T."
  hydrostatic=".F."
  phys_hydrostatic=".F."     # can be tested
  use_hydro_pressure=".F."   # can be tested
  consv_te="1."
else
  # hydrostatic options
  make_nh=".F."
  hydrostatic=".T."
  phys_hydrostatic=".F."     # will be ignored in hydro mode
  use_hydro_pressure=".T."   # have to be .T. in hydro mode
  consv_te="0."
fi

# variables for hyperthreading
if [ ${HYPT} = "on" ] ; then
  hyperthread=".true."
  div=2
else
  hyperthread=".false."
  div=1
fi
skip=`expr ${nthreads} \/ ${div}`

# when running with threads, need to use the following command
npes=`expr ${layout_x} \* ${layout_y} \* 6 `
LAUNCHER=${LAUNCHER:-srun}
if [ ${LAUNCHER} = 'srun' ] ; then
    export SLURM_CPU_BIND=verbose
    run_cmd="${LAUNCHER} --label --ntasks=$npes --cpus-per-task=$skip $CONTAINER ./${executable##*/}"
else
    export MPICH_ENV_DISPLAY=YES
    run_cmd="${LAUNCHER} -np $npes $CONTAINER ./${executable##*/}"
fi

# # set up the run area
# \rm -rf $WORKDIR
# mkdir -p $WORKDIR
# cd $WORKDIR
# mkdir -p RESTART
# mkdir -p INPUT

# copy over the executable
cp $executable .

# copy over the tables
#
# create an empty data_table
touch data_table

#
# build the diag_table with the experiment name and date stamp
cat > diag_table << EOF
${GRID}.${MODE}
0 0 0 0 0 0
"grid_spec",              -1,  "months",   1, "days",  "time"
"atmos_static",           -1,  "hours",    1, "hours", "time"
"atmos_daily",                 24, "hours",  1, "days",  "time"
"atmos_daily_ave",                 24, "hours",  1, "days",  "time"
"atmos_10day_ave",                 10, "days",  1, "days",  "time"
#output variables
#=======================
# ATMOSPHERE DIAGNOSTICS
#=======================
###
# grid_spec
###
 "dynamics", "grid_lon", "grid_lon", "grid_spec", "all", .false.,  "none", 2,
 "dynamics", "grid_lat", "grid_lat", "grid_spec", "all", .false.,  "none", 2,
 "dynamics", "grid_lont", "grid_lont", "grid_spec", "all", .false.,  "none", 2,
 "dynamics", "grid_latt", "grid_latt", "grid_spec", "all", .false.,  "none", 2,
 "dynamics", "area",     "area",     "grid_spec", "all", .false.,  "none", 2,
#### plevs
 "dynamics",  "h_plev",       "h_plev",      "atmos_daily",  "all",  .false.,  "none",  2
 "dynamics",  "u_plev",       "u_plev",      "atmos_daily",  "all",  .false.,  "none",  2
 "dynamics",  "v_plev",       "v_plev",      "atmos_daily",  "all",  .false.,  "none",  2
 "dynamics",  "t_plev",       "t_plev",      "atmos_daily",  "all",  .false.,  "none",  2
### misc. plevs
 "dynamics",  "vort850",        "VORT850",       "atmos_daily", "all", .false.,  "none", 2
 "dynamics",  "vort500",        "VORT500",       "atmos_daily", "all", .false.,  "none", 2
 "dynamics",  "vort200",        "VORT200",       "atmos_daily", "all", .false.,  "none", 2
 "dynamics",  "pv350K",         "pv350k",        "atmos_daily", "all", .false.,  "none", 2
 "dynamics",  "omg500",        "omg500",       "atmos_daily", "all", .false.,  "none", 2
#Layer-mean fields
 "dynamics", "t_plev_ave", "t_plev_ave", "atmos_daily",  "all",  .false.,  "none",  2
### Chemistry tracers
 "dynamics",  "acl",         "acl",        "atmos_daily", "all", .false.,  "none", 2
 "dynamics",  "acl2",        "acl2",       "atmos_daily", "all", .false.,  "none", 2
 "dynamics",  "acly",        "acly",       "atmos_daily", "all", .false.,  "none", 2
#### Integrated fields
 "dynamics",  "slp",         "PRMSL",      "atmos_daily", "all", .false.,  "none", 2
 "dynamics",  "ps",          "PRESsfc",     "atmos_daily", "all", .false., "none", 2
 "dynamics",  "tm",          "TMP500_300", "atmos_daily", "all", .false.,  "none", 2
###
# gfs static data
###
 "dynamics",      "pk",          "pk",           "atmos_static",      "all", .false.,  "none", 2
 "dynamics",      "bk",          "bk",           "atmos_static",      "all", .false.,  "none", 2
 "dynamics",      "hyam",        "hyam",         "atmos_static",      "all", .false.,  "none", 2
 "dynamics",      "hybm",        "hybm",         "atmos_static",      "all", .false.,  "none", 2
 "dynamics",      "zsurf",       "HGTsfc",          "atmos_static",      "all", .false.,  "none", 2
EOF


#
# build the field_table
cat > field_table <<EOF
# added by FRE: sphum must be present in atmos
# specific humidity for moist runs
 "TRACER", "atmos_mod", "sphum"
           "longname",     "specific humidity"
           "units",        "kg/kg"
       "profile_type", "fixed", "surface_value=1.e30" /
#idealized tracers
 "TRACER", "atmos_mod", "cl"
           "longname",     "cl"
           "units",        "kg/kg"
       "profile_type", "fixed", "surface_value=1.e30" /
#idealized tracers
 "TRACER", "atmos_mod", "cl2"
           "longname",     "cl2"
           "units",        "kg/kg"
       "profile_type", "fixed", "surface_value=1.e30" /
EOF


#
# build the field_table
cat > input.nml <<EOF
 &diag_manager_nml
    prepend_date = .F.
! this diag table creates a lot of files
! next three lines are necessary
    max_num_axis_sets = 100
    max_files = 100
    max_axes = 240
/
 &fms_affinity_nml
    affinity = .false.
/
 &fms_io_nml
       checksum_required   = .false.
       max_files_r = 100 
       max_files_w = 100 
/
 &fms_nml
       clock_grain = 'ROUTINE',
       domains_stack_size = 16000000,
       print_memory_usage = .false.
/
 &fv_core_nml
       layout   = $layout_x,$layout_y
       io_layout = $io_layout
       npx      = $npx
       npy      = $npy
       ntiles   = 6
       npz    = $npz
       npz_type = 'gfs'
       grid_type = 0
       make_nh = $make_nh
       fv_debug = .F.
       range_warn = .T.
       reset_eta = .F.
       n_sponge = 0
       nudge_qv = .F.
       RF_fast = .F.
       tau_h2o = 0.
       tau = -1
       rf_cutoff = -1
       d2_bg_k1 = 0.20
       d2_bg_k2 = 0.10 ! z2: increased
       kord_tm = -9
       kord_mt = 9
       kord_wz = 9
       kord_tr = 9
       hydrostatic = $hydrostatic
       phys_hydrostatic = $phys_hydrostatic
       use_hydro_pressure = $use_hydro_pressure
       beta = 0.
       a_imp = 1.
       p_fac = 0.05
       k_split = 1
       n_split = 16
       nwat = 0
       na_init = $na_init
       d_ext = 0.0
       dnats = 0
       d2_bg = 0.
       nord = 3
       dddmp = 0.5
       d4_bg = 0.15
       vtdm4 = 0.06
       delt_max = 0.002
       convert_ke = .F.
       ke_bg = 0.
       do_vort_damp = .T.
       external_ic = .F. !COLD START
       mountain = .F.
       hord_mt = 8
       hord_vt = 8
       hord_tm = 8
       hord_dp = 8
       hord_tr = 8 ! z2: changed
       adjust_dry_mass = .F.
       consv_te = 0.
       consv_am = .F.
       fill = .F.
       dwind_2d = .F.
       print_freq = 0
       warm_start = $warm_start
       is_ideal_case = .T.
       z_tracer = .T.
       fill_dp = .F.
       adiabatic = .F.
       do_Held_Suarez = .T.
/
&fv_diag_plevs_nml
    nplev=6
    levs = 150,200,300,500,850,1000
/
&test_case_nml ! cold start
    test_case = 14
/
 &main_nml
       days  = $days
       hours = $hours
       minutes = $minutes
       seconds = $seconds
       dt_atmos = $dt_atmos
       current_time =  $curr_date
       atmos_nthreads = $nthreads
       use_hyper_thread = $hyperthread
/
 &external_ic_nml
       filtered_terrain = .T.
       levp = 64
       gfs_dwinds = .T.
       checker_tr = .F.
       nt_checker = 0
/
 &gfdl_mp_nml
    do_qa = .false.
/
 &sim_phys_nml
  do_GFDL_sim_phys = .F.
  do_reed_sim_phys = .F.
/
 &reed_sim_phys_nml
    do_reed_cond = .F.
/
 &integ_phys_nml
       do_inline_mp = .T.
       do_sat_adj = .F.
/


EOF

# run the executable
${run_cmd} | tee fms.out || exit
num=`expr ${num} + 1`
echo "num=${num}" > ${RST_COUNT}

runnum=$(printf %04d $num)

ls ${WORKDIR}/rundir/RESTART


if [ ! -d ${WORKDIR}/ascii ] ; then
    mkdir -p ${WORKDIR}/ascii
fi

mkdir -p ${WORKDIR}/ascii/${runnum}


for out in *.out *.results input*.nml *_table 
do 
    mv ${out} ${WORKDIR}/ascii/${runnum}
done


cd ${WORKDIR}
resfile_list=${WORKDIR}/rundir/file.restart.list.txt
find ${WORKDIR}/rundir/RESTART -iname '*.res*' > $resfile_list

find ${WORKDIR}/rundir/RESTART -iname '*_data*' >> $resfile_list

resfiles=`wc -l ${resfile_list} | awk '{print $1}'`
echo ${resfiles}

#if [ ${resfiles} > 0 ] ; then 
restart_dir=${WORKDIR}/restart/${runnum}
#echo 
   
if [ ! -d ${restart_dir} ] ; then 
   mkdir -p ${restart_dir}
fi 
  
   
# xargs mv -t $restart_dir < $resfile_list # UNCOMMENT ME IF DOING SYMBOLIC LINK 
echo "restart_dir=${restart_dir}" >> ${RST_COUNT}
   
#fi 

cd ${WORKDIR}/rundir 


hist_dir=${WORKDIR}/history/${runnum}
if [ ! -d ${hist_dir} ] ; then 
     mkdir -p ${hist_dir}
fi 
   

for File in *.tile1.nc
do
    export variables=`ncdump -h $File | grep 'grid_yt, grid_xt' | awk '{print $2}' | cut -d\( -f1`
    export variables=`echo $variables |sed 's/ /,/g'`
    echo $variables
    export basename=${File%.*.*}
    echo $basename
    mpirun -np 2 fregrid --input_mosaic ${FV3_BUILDDIR}/../grid_files/C96_grid_files/C96_mosaic.nc \
                                 --input_file $basename --interp_method conserve_order2 \
                                 --remap_file fregrid_remap_file --nlon 288 --nlat 180 --scalar_field $variables
    echo ${basename}.nc 'interpolated to lat-lon'
    mv ${basename}.nc ${hist_dir}/
done

if [[ ${num} < ${NUM_TOT} ]] ; then
  echo "resubmitting... "
    cd $RUN_DIR
    echo `pwd`
    echo ${SCRIPT##*/}
    ./${SCRIPT##*/}
fi



   


#
# verification
# < add logic for verification >
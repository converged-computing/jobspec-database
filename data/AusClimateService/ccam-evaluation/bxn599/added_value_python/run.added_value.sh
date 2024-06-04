#!/bin/bash
#PBS -N av-pav-rav
#PBS -P q49
#PBS -q normalbw
#PBS -l walltime=12:00:00
#PBS -l mem=128gb
#PBS -l ncpus=14
#PBS -l storage=gdata/hh5+gdata/q49+scratch/oi10+gdata/zv2+gdata/rr3+gdata/ma05+gdata/r87+gdata/ub4+gdata/tp28+scratch/e53+scratch/e53
#PBS -l jobfs=100GB

# User specific aliases and functions
export PATH="~/bin/:$PATH"
export PYTHONPATH="${PYTHONPATH}:/g/data/xv83/users/bxn599/CaRSA/example_rav_cs/"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1


export DASK_DISTRIBUTED__SCHEDULER__ALLOWED_FAILURES=20
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT=60
export DASK_DISTRIBUTED__COMM__RETRY__COUNT=5


module use /g/data3/hh5/public/modules
module load conda/analysis3-21.01 #unstable with xesmf conervative_normed and intel compiler


###
# Init

#< Where are the original scripts
path=/g/data/xv83/users/bxn599/CaRSA/example_rav_cs/
#< Where to save all the output
outpath=/g/data/xv83/users/bxn599/CaRSA/example_rav_cs/

#< Resolution
res=0p11deg

#< Common settings
quantiles=mean,0.99
seasons="annual,DJF"
interp_method="linear"



###############################
# Create the output directories
mkdir -p ${outpath}
mkdir -p ${outpath}/scripts
mkdir -p ${outpath}/plots
mkdir -p ${outpath}/data

scriptpath=${outpath}/scripts/
plotpath=${outpath}/plots/
datapath=${outpath}/data/

# Copy the scripts used to the output directory
cp ${path}/run.added_value.sh ${scriptpath}
cp ${path}/dev.added_value.py ${scriptpath}
cp ${path}/dev.potential_added_value.py ${scriptpath}
cp ${path}/dev.realised_added_value.py ${scriptpath}
cp ${path}/dev.plot_added_value_overview.py ${scriptpath}


###
# Main

################################################################################################################################################################################################################################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################################################################################################################################################################################################################################
echo @@@@@@@@@@@@@@@@@@@
echo CCAM
echo @@@@@@@@@@@@@@@@@@@
##
# CCAM - ACCESS
project=CCAM
gdd=access1-0
yrStartPast=1986
yrEndPast=2005
yrStartFut=2080
yrEndFut=2099


###
#< Maximum temperature
echo !!!!!!!!!!!!!!!!!!!
echo MAXIMUM TEMPERATURE
echo !!!!!!!!!!!!!!!!!!!
ifile_gdd_hist=/g/data/rr3/publications/CMIP5/output1/CSIRO-BOM/ACCESS1-0/historical/day/atmos/day/r1i1p1/v20131108/tasmax/tasmax*.nc
ifile_gdd_fut=/g/data/rr3/publications/CMIP5/output1/CSIRO-BOM/ACCESS1-0/rcp85/day/atmos/day/r1i1p1/latest/tasmax/tasmax*.nc
ifile_rcm=/scratch/e53/mxt599/esci/ccam_access1-0_aus_12km/regrid/tasmax*.nc
ifile_obs=/g/data/zv2/agcd/v1/tmax/mean/r005/01day/*.nc
varname_gdd=tasmax
varname_rcm=tasmax
varname_obs=tmax
outunit="K"
prefix=${project}_${gdd}.${varname_rcm}.${res}
varnames=${varname_gdd},${varname_rcm},${varname_obs}

python -W ignore $path/dev.added_value.py --ifile_gdd ${ifile_gdd_hist} --ifile_rcm ${ifile_rcm} --ifile_obs ${ifile_obs} --varnames ${varnames} --outunit ${outunit} --yrStart ${yrStartPast} --yrEnd ${yrEndPast} --quantiles ${quantiles} --seasons ${seasons} -o ${datapath} --prefix ${prefix} --interp_method ${interp_method}  || exit
python -W ignore $path/dev.potential_added_value.py --ifile_gdd ${ifile_gdd_hist},${ifile_gdd_fut} --ifile_rcm ${ifile_rcm} --varnames ${varnames} --outunit ${outunit} --yrStartPast ${yrStartPast} --yrEndPast ${yrEndPast} --yrStartFut ${yrStartFut} --yrEndFut ${yrEndFut} --quantiles ${quantiles} --seasons ${seasons} -o ${datapath} --prefix ${prefix} --interp_method ${interp_method}  || exit
python -W ignore $path/dev.realised_added_value.py --ipath ${datapath} --ifile_obs ${ifile_obs} --varname_obs ${varname_obs} --yrStartPast ${yrStartPast} --yrEndPast ${yrEndPast} --yrStartFut ${yrStartFut} --yrEndFut ${yrEndFut} --quantiles ${quantiles} --seasons ${seasons} -o ${datapath} --prefix ${prefix} --interp_method ${interp_method}  || exit
python -W ignore $path/dev.plot_added_value_overview.py --ipath ${datapath} --varname ${varname_rcm} --yrStartPast ${yrStartPast} --yrEndPast ${yrEndPast} --yrStartFut ${yrStartFut} --yrEndFut ${yrEndFut} --quantiles ${quantiles} --seasons ${seasons} -o ${plotpath} --prefix ${prefix} --interp_method ${interp_method}  || exit

###
#< Minimum temperature
echo !!!!!!!!!!!!!!!!!!!
echo MINIMUM TEMPERATURE
echo !!!!!!!!!!!!!!!!!!!
ifile_gdd_hist=/g/data/rr3/publications/CMIP5/output1/CSIRO-BOM/ACCESS1-0/historical/day/atmos/day/r1i1p1/v20131108/tasmin/tasmin*.nc
ifile_gdd_fut=/g/data/rr3/publications/CMIP5/output1/CSIRO-BOM/ACCESS1-0/rcp85/day/atmos/day/r1i1p1/latest/tasmin/tasmin*.nc
ifile_rcm=/scratch/e53/mxt599/esci/ccam_access1-0_aus_12km/regrid/tasmin*.nc
ifile_obs=/g/data/zv2/agcd/v1/tmin/mean/r005/01day/*.nc
varname_gdd=tasmin
varname_rcm=tasmin
varname_obs=tmin
outunit="K"
prefix=${project}_${gdd}.${varname_rcm}.${res}
varnames=${varname_gdd},${varname_rcm},${varname_obs}

python -W ignore $path/dev.added_value.py --ifile_gdd ${ifile_gdd_hist} --ifile_rcm ${ifile_rcm} --ifile_obs ${ifile_obs} --varnames ${varnames} --outunit ${outunit} --yrStart ${yrStartPast} --yrEnd ${yrEndPast} --quantiles ${quantiles} --seasons ${seasons} -o ${datapath} --prefix ${prefix} --interp_method ${interp_method}  || exit
python -W ignore $path/dev.potential_added_value.py --ifile_gdd ${ifile_gdd_hist},${ifile_gdd_fut} --ifile_rcm ${ifile_rcm} --varnames ${varnames} --outunit ${outunit} --yrStartPast ${yrStartPast} --yrEndPast ${yrEndPast} --yrStartFut ${yrStartFut} --yrEndFut ${yrEndFut} --quantiles ${quantiles} --seasons ${seasons} -o ${datapath} --prefix ${prefix} --interp_method ${interp_method}  || exit
python -W ignore $path/dev.realised_added_value.py --ipath ${datapath} --ifile_obs ${ifile_obs} --varname_obs ${varname_obs} --yrStartPast ${yrStartPast} --yrEndPast ${yrEndPast} --yrStartFut ${yrStartFut} --yrEndFut ${yrEndFut} --quantiles ${quantiles} --seasons ${seasons} -o ${datapath} --prefix ${prefix} --interp_method ${interp_method}  || exit
python -W ignore $path/dev.plot_added_value_overview.py --ipath ${datapath} --varname ${varname_rcm} --yrStartPast ${yrStartPast} --yrEndPast ${yrEndPast} --yrStartFut ${yrStartFut} --yrEndFut ${yrEndFut} --quantiles ${quantiles} --seasons ${seasons} -o ${plotpath} --prefix ${prefix} --interp_method ${interp_method}  || exit


###
#< Precip
echo !!!!!!!!!!!!!!!!!!!
echo PRECIPITATION
echo !!!!!!!!!!!!!!!!!!!
ifile_gdd_hist=/g/data/rr3/publications/CMIP5/output1/CSIRO-BOM/ACCESS1-0/historical/day/atmos/day/r1i1p1/v20131108/pr/pr*.nc
ifile_gdd_fut=/g/data/rr3/publications/CMIP5/output1/CSIRO-BOM/ACCESS1-0/rcp85/day/atmos/day/r1i1p1/latest/pr/pr*.nc
ifile_rcm=/scratch/e53/mxt599/esci/ccam_access1-0_aus_12km/regrid/pr*.nc
ifile_obs=/g/data/zv2/agcd/v1/precip/calib/r005/01day/*.nc
varname_gdd=pr
varname_rcm=pr
varname_obs=precip
outunit="mm day-1"
prefix=${project}_${gdd}.${varname_rcm}.${res}
varnames=${varname_gdd},${varname_rcm},${varname_obs}

python -W ignore $path/dev.added_value.py --ifile_gdd ${ifile_gdd_hist} --ifile_rcm ${ifile_rcm} --ifile_obs ${ifile_obs} --varnames ${varnames} --outunit ${outunit} --yrStart ${yrStartPast} --yrEnd ${yrEndPast} --quantiles ${quantiles} --seasons ${seasons} -o ${datapath} --prefix ${prefix} --interp_method ${interp_method}  || exit
python -W ignore $path/dev.potential_added_value.py --ifile_gdd ${ifile_gdd_hist},${ifile_gdd_fut} --ifile_rcm ${ifile_rcm} --varnames ${varnames} --outunit ${outunit} --yrStartPast ${yrStartPast} --yrEndPast ${yrEndPast} --yrStartFut ${yrStartFut} --yrEndFut ${yrEndFut} --quantiles ${quantiles} --seasons ${seasons} -o ${datapath} --prefix ${prefix} --interp_method ${interp_method}  || exit
python -W ignore $path/dev.realised_added_value.py --ipath ${datapath} --ifile_obs ${ifile_obs} --varname_obs ${varname_obs} --yrStartPast ${yrStartPast} --yrEndPast ${yrEndPast} --yrStartFut ${yrStartFut} --yrEndFut ${yrEndFut} --quantiles ${quantiles} --seasons ${seasons} -o ${datapath} --prefix ${prefix} --interp_method ${interp_method}  || exit
python -W ignore $path/dev.plot_added_value_overview.py --ipath ${datapath} --varname ${varname_rcm} --yrStartPast ${yrStartPast} --yrEndPast ${yrEndPast} --yrStartFut ${yrStartFut} --yrEndFut ${yrEndFut} --quantiles ${quantiles} --seasons ${seasons} -o ${plotpath} --prefix ${prefix} --interp_method ${interp_method}  || exit


###
# added value boxplot summary
echo !!!!!!!!!!!!!!!!!!!
echo BOX PLOTS
echo !!!!!!!!!!!!!!!!!!!

gdds=access1-0
prefix_boxplot=${project}_${gdd}
regions=/g/data/xv83/users/bxn599/CaRSA/example_rav_cs/added_value.regions.barpa.${res}.remove_west_boundary.nc
varnames=tasmax,tasmin,pr
python -W ignore $path/dev.added_value_boxplot.py --regions ${regions} --project ${project} --prefix ${prefix_boxplot} --ipath ${datapath} --resolution ${res} --op mse --gdds ${gdds} --varnames ${varnames} --yrStart ${yrStartPast} --yrEnd ${yrEndPast} --quantiles ${quantiles} --seasons ${seasons} -o ${plotpath} --interp_method ${interp_method}


###
# realised added value boxplot summary
l_new=True
varnames=tasmax,tasmin,pr
varnames_obs=tmax,tmin,precip
#ifiles_obs=/g/data/rr8/OBS/AWAP_ongoing/v0.4/grid_05/daily/tmax/tmax_mean_0.05_*.nc,/g/data/rr8/OBS/AWAP_ongoing/v0.4/grid_05/daily/tmin/tmin_mean_0.05_*.nc,/g/data/rr8/OBS/AWAP_ongoing/v0.4/grid_05/daily/precip/precip_calib_0.05_*.nc
ifiles_obs=/g/data/zv2/agcd/v1/tmax/mean/r005/01day/*.nc,/g/data/zv2/agcd/v1/tmin/mean/r005/01day*.nc,/g/data/zv2/agcd/v1/precip/calib/r005/01day/*.nc
python -W ignore $path/dev.rav_boxplot.py --regions ${regions} --project ${project} --l_new ${l_new} --prefix ${prefix_boxplot} --normalise True --resolution ${res} --ipath ${datapath} --ifiles_obs ${ifiles_obs} --op mse --gdds ${gdds} --varnames ${varnames} --varnames_obs ${varnames_obs} --yrStartPast ${yrStartPast} --yrEndPast ${yrEndPast} --yrStartFut ${yrStartFut} --yrEndFut ${yrEndFut} --quantiles ${quantiles} --seasons ${seasons} -o ${plotpath} --interp_method ${interp_method}

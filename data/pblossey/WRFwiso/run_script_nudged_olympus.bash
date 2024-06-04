#!/bin/bash
#PBS -N AntarcWISO_preprocess
#PBS -A UWAS0052
#PBS -l walltime=12:00:00
#PBS -q economy
#PBS -j oe
#PBS -m e
#PBS -M pblossey@uw.edu
#PBS -l select=1:ncpus=1:mpiprocs=1

# set to 2 for 45km outer domain with a 15km nest
# set to 3 to add a 5km nest over West Antarctica
NUM_DOMAINS=2

SYY=2015
SMM=1
SDD=31

EYY=2015
EMM=2
EDD=2

LAT_MIN=-100
LAT_MAX=-30
LON_MIN=-400
LON_MAX=400

TODAY=`date --iso-8601`

RUNNAME=nudge_1deg_llnl_base
H0TAG=2015-2017

BASEDIR=/home/disk/eos1/bloss/Runs/WAIS/Devel/WRFwiso-2019-01-03/

### Set TMPDIR as recommended
#export TMPDIR=/glade/scratch/$USER/temp
#mkdir -p $TMPDIR

cd $BASEDIR

# load modules for intel/netcdf setup
#module load matlab nco

# load modules for intel/netcdf setup
source /etc/profile.d/modules.sh
module load intel/13.1.1 openmpi/1.6.4 netcdf/4.3.0

# clean up folder with WPS Intermediate format files from CESM
pushd ConvertGCMOutputToWPSFormat
if [ ! -d Output ]; then
  mkdir Output
else
  rm -f Output/*
fi
popd

# Run matlab script to pull data from CESM output and put it into a netcdf equivalent of the WPS intermediate format
pushd ConvertGCMOutputToWPSFormat/CESM/Matlab

if [ ! -d GCMOutput ]; then
  echo " ***** Error in ConvertGCMOutputToWPSFormat/CESM/Matlab ****** "
  echo " ***** Create GCMOutput symbolic link to CESM data  ********** "
  exit 9
fi

SCRIPTNAME=matlab_script_$(printf %04d $SYY)_$(printf %02d $SMM)_$(printf %02d $SDD).m
echo ${SCRIPTNAME}
rm -f ${SCRIPTNAME}

echo " ***** Convert from CESM output to WPS-ready netcdf ***** "
for file in `ls GCMOutput/${RUNNAME}.cam.h1.$(printf %04d $SYY)-$(printf %02d $SMM)*.nc`
do
  file2=`echo ${file} | sed "s/GCMOutput\///g; s/@//g"`
  echo ${file} ${file2}
  echo "CESM2WPSNetcdf('CESM','${RUNNAME}','${H0TAG}','${file2}',${LAT_MIN},${LAT_MAX},${LON_MIN},${LON_MAX},${SYY},${SMM},${SDD},${EYY},${EMM},${EDD})" >> ${SCRIPTNAME}
##  matlab -nodesktop -nosplash -nodisplay -r "CESM2WPSNetcdf('CESM','${RUNNAME}','${H0TAG}','${file2}',${LAT_MIN},${LAT_MAX},${LON_MIN},${LON_MAX},${SYY},${SMM},${SDD},${EYY},${EMM},${EDD}); exit" &>> log.matlab
##  matlab -nodesktop -nosplash -nodisplay -r "CESM2WPSNetcdf('CESM','${RUNNAME}','${H0TAG}','${file2}',${SYY},${SMM},${SDD},${EYY},${EMM},${EDD}); exit" &> log.matlab
done

if [ ! $SYY -eq $EYY ] || [ ! $SMM -eq $EMM ]
then
    for file in `ls GCMOutput/*.cam.h1.$(printf %04d $EYY)-$(printf %02d $EMM)*.nc`
    do
	file2=`echo ${file} | sed "s/GCMOutput\///g; s/@//g"`
	echo ${file} ${file2}
  echo "CESM2WPSNetcdf('CESM','${RUNNAME}','${H0TAG}','${file2}',${LAT_MIN},${LAT_MAX},${LON_MIN},${LON_MAX},${SYY},${SMM},${SDD},${EYY},${EMM},${EDD})" >> ${SCRIPTNAME}
##	matlab -nodesktop -nosplash -nodisplay -r "CESM2WPSNetcdf('CESM','${RUNNAME}','${H0TAG}','${file2}',${LAT_MIN},${LAT_MAX},${LON_MIN},${LON_MAX},${SYY},${SMM},${SDD},${EYY},${EMM},${EDD}); exit" &>> log.matlab
##	matlab -nodesktop -nosplash -nodisplay -r "CESM2WPSNetcdf('CESM','${RUNNAME}','${H0TAG}','${file2}',${SYY},${SMM},${SDD},${EYY},${EMM},${EDD}); exit" &> log.matlab
    done
fi

matlab -nodesktop -nosplash -nodisplay -r "try run('${SCRIPTNAME}'); catch exit; end; exit" &>> log.${SCRIPTNAME}

popd

# compile the fortran program that converts this WPS-ready netcdf into binary WPS intermediate format
pushd ConvertGCMOutputToWPSFormat/Fortran
echo " ***** Making conversion program ***** "
make clean; make all
popd

# convert the WPS-ready netcdf into binary WPS intermediate format
pushd ConvertGCMOutputToWPSFormat/Output
echo " ***** Converting to Intermediate Format ***** "
for file in `ls *.nc`; do nice ../Fortran/WPSNetcdf2IntermediateFormat ${file}; done
popd

# move into the WPS directory
pushd WPS/

# set up netcdf library path for metgrid
# TODO: set up configure.wps and configure.wrf with -Wl,-rpath so that
#   we do not need to play these games with the LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/modules/netcdf/4.3.0/intel/13.1.1/lib

# link to the CESM data in WPS Intermediate Format
rm -f CESM*
ln -sf ../ConvertGCMOutputToWPSFormat/Output/CESM* .

# Use the Antarctica namelist
rm -f namelist.wps
sed "s/SYY/$(printf %04d $SYY)/g; s/EYY/$(printf %04d $EYY)/g; s/SMM/$(printf %02d $SMM)/g; s/EMM/$(printf %02d $EMM)/g; s/SDD/$(printf %02d $SDD)/g; s/EDD/$(printf %02d $EDD)/g; s/NUM_DOMAINS/${NUM_DOMAINS}/g" namelist.wps.Antarctica.45km15km5km.base > namelist.wps

# make sure that geogrid.exe has already been run
if [ ! -f geo_em.d01.nc ]; then
  echo " ***** Error in WPS                                    ***** "
  echo " ***** Run geogrid.exe before run_script_cheyenne.bash ***** "
  exit 9
fi

# run metgrid
echo " ***** Running Metgrid ***** "
nice ./metgrid.exe # &> log.metgrid 

# remove the netcdf path, so that nco will run properly
export LD_LIBRARY_PATH=/usr/local/intel/composer_xe_2013.3.163/compiler/lib/intel64

# recompute the sea ice, ice depth and snow-on-sea-ice amounts to
#   translate from the CESM world where land, ocean and sea ice share
#   grid cells to the WRF world where all ocean/sea ice grid cells are
#   free of land.  Also, CESM outputs for ice depth and
#   snow-on-sea-ice are weighted by ice area, so that compute the mean
#   values for locations with sea ice.
for file in `ls met_em*.nc`
   do 
    ncap2 -s "SEAICE=SIFRAC/(SIFRAC+OCNFRAC+1.e-12);ICEDEPTH=WGTSIDPTH/(SIFRAC+1.e-12);SNOWSI=WGTSNOWSI/(SIFRAC+1.e-12)" ${file} tmp.nc
     mv -f tmp.nc ${file}
     ncatted -a FLAG_ICEDEPTH,global,c,i,1 -a FLAG_SNOWSI,global,c,i,1 ${file}
done

popd

pushd WRFV3/test/em_real/

# Use the Antarctica namelist
rm -f namelist.input
sed "s/SYY/$(printf %04d $SYY)/g; s/EYY/$(printf %04d $EYY)/g; s/SMM/$(printf %02d $SMM)/g; s/EMM/$(printf %02d $EMM)/g; s/SDD/$(printf %02d $SDD)/g; s/EDD/$(printf %02d $EDD)/g; s/NUM_DOMAINS/${NUM_DOMAINS}/g" namelist.input.AntarcticaWISO.45km15km5km.base > namelist.input

ln -sf ../../../WPS/met_em* .

### Run the executable
mpiexec_mpt dplace -s 1 ./real.exe > log_real_${TODAY}

popd

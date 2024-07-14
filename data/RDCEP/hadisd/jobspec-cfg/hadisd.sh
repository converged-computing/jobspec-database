#!/bin/bash

CDO='cdo -f nc4 -z zip'
# FILE='hadisd.1.0.0.2011f.501360-99999.nc4'
set -x
FILE=$1

pushd hadisd2012

STATION=${FILE%.*}
mkdir $STATION
pushd $STATION

CDO segfaults when combining setname with the others
${CDO} daymax -selname,temperatures ../${FILE} tmax.nc4.1
${CDO} setname,tmax tmax.nc4.1 tmax.nc4; rm tmax.nc4.1
${CDO} daymin -selname,temperatures ../${FILE} tmin.nc4.1 
${CDO} setname,tmin tmin.nc4.1 tmin.nc4; rm tmin.nc4.1
${CDO} daysum -selname,precip1_depth ../${FILE} precip.nc4.1
${CDO} setname,precip precip.nc4.1 precip.nc4; rm precip.nc4.1
${CDO} daymean -selname,windspeeds ../${FILE} wind.nc4.1
${CDO} setname,wind wind.nc4.1 wind.nc4; rm wind.nc4.1
${CDO} daymean -selname,dewpoints ../${FILE} dewp.nc4.1
${CDO} setname,dewp dewp.nc4.1 dewp.nc4; rm dewp.nc4.1
${CDO} merge {tmax,tmin,precip,wind,dewp}.nc4 merge.nc4.1
${CDO} setreftime,1973-01-01,00:00:00,days merge.nc4.1 merge.nc4; rm merge.nc4.1
../../hadisd.R

popd
popd

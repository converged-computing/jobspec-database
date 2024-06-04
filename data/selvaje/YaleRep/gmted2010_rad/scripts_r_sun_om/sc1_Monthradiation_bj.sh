# cd  /mnt/data2/scratch/GMTED2010/grassdb 
# ls  /mnt/data2/scratch/GMTED2010/tiles/mn75_grd_tif/?_?.tif  | xargs -n 1  -P 10  bash /mnt/data2/scratch/GMTED2010/scripts/sc2_real-sky-horiz-solar_Monthradiation.sh
# ls  /mnt/data2/scratch/GMTED2010/tiles/mn75_grd_tif/{1_1,1_2,2_1,2_2,1_0,2_0}.tif  | xargs -n 1  -P 10  bash /mnt/data2/scratch/GMTED2010/scripts/sc2_real-sky-horiz-solar_Monthradiation.sh

# alaska and maine 
# ls  /mnt/data2/scratch/GMTED2010/tiles/mn75_grd_tif/{0_0,3_1}.tif  | xargs -n 1  -P 10  bash /mnt/data2/scratch/GMTED2010/scripts/sc2_real-sky-horiz-solar_Monthradiation.sh
# ls  /mnt/data2/scratch/GMTED2010/tiles/mn75_grd_tif/2_1.tif  | xargs -n 1  -P 10  bash /mnt/data2/scratch/GMTED2010/scripts/sc2_real-sky-horiz-solar_Monthradiation.sh
# ls  /mnt/data2/scratch/GMTED2010/tiles/mn75_grd_tif/2_2.tif  | xargs -n 1  -P 10  bash /mnt/data2/scratch/GMTED2010/scripts/sc2_real-sky-horiz-solar_Monthradiation.sh


# for tile  in $( awk '{ if (NR>1) print $1 }' /lustre/scratch/client/fas/sbsc/ga254/dataproces/GMTED2010/geo_file/tile_lat_long_20d.txt ) ;  do  qsub -v tile=$tile /home/fas/sbsc/ga254/scripts/gmted2010_rad/scripts_r_sun_om/sc1_Monthradiation_bj.sh  ; done 
# for tile  in $( awk '{ if (NR>1) print $1 }' /lustre/scratch/client/fas/sbsc/ga254/dataproces/GMTED2010/geo_file/tile_lat_long_20d.txt ) ;  do  bash  /home/fas/sbsc/ga254/scripts/gmted2010_rad/scripts_r_sun_om/sc1_Monthradiation_bj.sh $tile   ; done 

# un tile ha impiegato 6 ore

#PBS -S /bin/bash 
#PBS -q fas_normal
#PBS -l walltime=1:00:00:00  
#PBS -l nodes=1:ppn=1
#PBS -V
#PBS -o  /scratch/fas/sbsc/ga254/stdout 
#PBS -e  /scratch/fas/sbsc/ga254/stderr

# export tile=$1

export tile=$tile

export INTIF_mn30=/lustre/scratch/client/fas/sbsc/ga254/dataproces/GMTED2010/tiles/mn30_grd_tif
export INTIF=/lustre/scratch/client/fas/sbsc/ga254/dataproces/GMTED2010/tiles/be75_grd_tif
export INTIFL=/lustre/scratch/client/fas/sbsc/ga254/dataproces/GMTED2010/tiles/
export INDIRG=/lustre/scratch/client/fas/sbsc/ga254/dataproces/SOLAR/grassdb
export OUTDIR=/lustre/scratch/client/fas/sbsc/ga254/dataproces/SOLAR/radiation
export RAMDIRG=/dev/shm/

# echo clip the data 

cd $INDIRG 

ulx=$(grep $tile  /lustre/scratch/client/fas/sbsc/ga254/dataproces/GMTED2010/geo_file/tile_lat_long_20d.txt  | awk '{  print  $4 - 1 }')
uly=$(grep $tile  /lustre/scratch/client/fas/sbsc/ga254/dataproces/GMTED2010/geo_file/tile_lat_long_20d.txt  | awk '{  print  $5 + 1 }')
lrx=$(grep $tile  /lustre/scratch/client/fas/sbsc/ga254/dataproces/GMTED2010/geo_file/tile_lat_long_20d.txt  | awk '{  print  $6 + 1 }')
lry=$(grep $tile  /lustre/scratch/client/fas/sbsc/ga254/dataproces/GMTED2010/geo_file/tile_lat_long_20d.txt  | awk '{  print  $7 - 1 }')

if [ $ulx -lt -180  ] ; then  ulx=-180 ; fi
if [ $uly -gt  90   ] ; then  uly=90   ; fi 
if [ $lrx -gt  180  ] ; then  lrx=180  ; fi 
if [ $lry -lt -90   ] ; then  lry=-90  ; fi

# gdal_edit.py  -a_ullr -180 84 +180 -90       /lustre/scratch/client/fas/sbsc/ga254/dataproces/GMTED2010/tiles/mn30_grd_tif/mn30_grd.tif  to enforce exact tif border 
gdal_translate -co  COMPRESS=LZW -co ZLEVEL=9  -projwin $ulx $uly $lrx $lry  $INTIF_mn30/mn30_grd.tif $INDIRG/$tile.tif 

max=$(pkinfo  -mm   -i  $INDIRG/$tile.tif   | awk '{  print $4 }')

if [ $max -eq 0 ] ; then 
    exit 
else 

# gdal_translate -srcwin 9000 4200 50 50  $INTIF_mn30/mn30_grd.tif $tile.tif  # to create a file test 

echo create location  loc_$tile 

rm -rf  /dev/shm/*
mkdir -p  $RAMDIRG/loc_tmp/tmp

echo "LOCATION_NAME: loc_tmp"                                                       > $HOME/.grass7/rc_$tile
echo "GISDBASE: /dev/shm"                                                          >> $HOME/.grass7/rc_$tile
echo "MAPSET: tmp"                                                                 >> $HOME/.grass7/rc_$tile
echo "GRASS_GUI: text"                                                             >> $HOME/.grass7/rc_$tile

# path to GRASS settings file
export GISRC=$HOME/.grass7/rc_$tile
export GRASS_PYTHON=python
export GRASS_MESSAGE_FORMAT=plain
export GRASS_PAGER=cat
export GRASS_WISH=wish
export GRASS_ADDON_BASE=$HOME/.grass7/addons
export GRASS_VERSION=7.0.0beta1
export GISBASE=/usr/local/cluster/hpc/Apps/GRASS/7.0.0beta1/grass-7.0.0beta1
export GRASS_PROJSHARE=/usr/local/cluster/hpc/Libs/PROJ/4.8.0/share/proj/
export PROJ_DIR=/usr/local/cluster/hpc/Libs/PROJ/4.8.0

export PATH="$GISBASE/bin:$GISBASE/scripts:$PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$GISBASE/lib"
export GRASS_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export PYTHONPATH="$GISBASE/etc/python:$PYTHONPATH"
export MANPATH=$MANPATH:$GISBASE/man
export GIS_LOCK=$$
export GRASS_OVERWRITE=1

rm -rf  $RAMDIRG/loc_$tile

echo start importing 
r.in.gdal in=$INDIRG/$tile.tif   out=$tile  location=loc_$tile

g.mapset mapset=PERMANENT  location=loc_$tile

echo import and set the mask 
r.external  -o input=/lustre/scratch/client/fas/sbsc/ga254/dataproces/GSHHG/GSHHS_tif_1km/land_mask_m10fltGSHHS_f_L1_buf10.tif   output=mask  --overwrite  --quiet 
r.mask raster=mask

# echo calculate r.slope.aspect  # suppressed in case of horizontal surface
# r.slope.aspect elevation=$tile    aspect=aspect_$tile  slope=slope_$tile   # for horizontal surface not usefull

echo calculate horizon

# step 1 in r.sun = horizonstep=15   360/24 = 15       # for 30 min 7.5

export hstep=7.5

r.horizon  elevin=$tile     horizonstep=$hstep  horizon=horiz   maxdistance=200000


# setting the mask at validation point level usefull for point validation 
# r.external  -o input=/lustre/scratch/client/fas/sbsc/ga254/dataproces/SOLAR/shp_in/point_nsrdb.tif   output=mask_point   --overwrite  --quiet
# r.mask raster=mask_point  --o 


for monthc in  01 02 03 04 05 06 07 08 09 10 11 12 ; do 

daystart=$(awk -v monthc=$monthc '{ if (substr($1,0,2)==monthc) print $2  }'  /lustre/scratch/client/fas/sbsc/ga254/dataproces/MERRAero/tif_day/MMDD_JD_0JD.txt  | head -1)
dayend=$(awk   -v monthc=$monthc '{ if (substr($1,0,2)==monthc) print $2  }'  /lustre/scratch/client/fas/sbsc/ga254/dataproces/MERRAero/tif_day/MMDD_JD_0JD.txt  | tail -1)

export monthc=$monthc

seq $daystart $dayend   | xargs -n 1  -P 8  bash  -c $'

day=$1

echo start the computation for day $day 

# the albedo influence only the reflectance radiation 
# echo  import albedo 
# r.external  -o input=/lustre/scratch/client/fas/sbsc/ga254/dataproces/MODALB/0.3_5.0.um.00-04.WS.c004.v2.0/AlbMap.WS.c004.v2.0.00-04.${dayi}.0.3_5.0.tif   output=albedo${day}_${tile} --overwrite --quiet
# r.mapcalc  " albedo${day}_${tile}_coef =  albedo${day}_${tile}  * 0.001" 

echo  import cloud
# for this case import the same tif  for the full month
# coef_bh 1 no cloud 0 full cloud 

r.external -o input=/lustre/scratch/client/fas/sbsc/ga254/dataproces/CLOUD/day_estimation_linear/cloud${day}.tif  output=cloud${day}_${tile}  --overwrite --quiet
r.mapcalc  " cloud${day}_${tile}_coef = 1 -  ( cloud${day}_${tile} * 0.0001)" 

echo import Aerosol 
# coef_dh 1 no Aerosol 0 full Aerosol  
r.external -o input=/lustre/scratch/client/fas/sbsc/ga254/dataproces/AE_C6_MYD04_L2/temp_smoth_1km/AOD_1km_day${day}.tif  output=aeros${day}_${tile}  --overwrite --quiet

# see http://en.wikipedia.org/wiki/Optical_depth 
# 2.718281828^−(5000/1000) = 0.006737947
# 2.718281828^−(−50/1000) = 1.051271096 
# the animation formula is the following 1 - (  2.718281828^(- (aeros${day}_${tile} * 0.001))) "
# for the coef_df take out -1 

r.mapcalc " aeros${day}_${tile}_coef =  1 -  (  2.718281828^(- (aeros${day}_${tile} * 0.001))) "

# horizontal clear sky 
# take out aspin=aspect_$tile  slopein=slope_$tile to simulate horizontal behaviur    better specify the slope=0 
# in case of horizontal behaviur the reflectance is = 0 
# also the albedo dose not influence the  global radiation . Better say the albedo influence only the reflectance radiation and it isualy a very small value - between 0 and 1 -  for the horizontal surface 
# in case of not horizontal surface the reflectance paly an important role. 
# indead the glob_HradT_day${day}_month${monthc} = glob_HraAl_day${day}_month${monthc}

# beam_rad e influenzato solo dal Cloud
# diff_rad e influenzato solo dal AOD

# make anonther test with inclined surface

# horizontal aerosol and cloud
# transparent = T

r.sun  --o   elev_in=$tile   \
lin=3   \
day=$day  horizon=horiz  horizonstep=$hstep   --overwrite  \
diff_rad=diff_HradT_day${day}_month${monthc} \
beam_rad=beam_HradT_day${day}_month${monthc} --q
## glob_rad=glob_HradT_day${day}_month${monthc} \   # is real the sum of diff beam and rad so no calculation 
##  refl_rad=refl_HradT_day${day}_month${monthc}     # orizontal 0 reflectance 

r.out.gdal -c type=Float32  nodata=-1  createopt="COMPRESS=LZW,ZLEVEL=9"    input=diff_HradT_day${day}_month${monthc}   output=$OUTDIR/diff_Hrad_day_tiles/$day/diff_HradT_day$day"_"month$monthc"_"$tile.tif --q --o
r.out.gdal -c type=Float32  nodata=-1  createopt="COMPRESS=LZW,ZLEVEL=9"    input=beam_HradT_day${day}_month${monthc}   output=$OUTDIR/beam_Hrad_day_tiles/$day/beam_HradT_day$day"_"month$monthc"_"$tile.tif --q --o
r.out.gdal -c type=Float32  nodata=-1  createopt="COMPRESS=LZW,ZLEVEL=9"  input=refl_HradT_day${day}_month${monthc}   output=$OUTDIR/refl_Hrad_day_tiles/$day/refl_HradT_day$day"_"month$monthc"_"$tile.tif --q --o 

# CLOUD AOD 

r.sun  --o   elev_in=$tile     \
lin=3  \
day=$day   horizon=horiz  horizonstep=$hstep  --overwrite  \
coef_bh=cloud${day}_${tile}_coef  coef_dh=aeros${day}_${tile}_coef \
diff_rad=diff_HradCA_day${day}_month${monthc} \
beam_rad=beam_HradCA_day${day}_month${monthc} --q 
# glob_rad=glob_HradCA_day${day}_month${monthc} \
# refl_rad=refl_HradCA_day${day}_month${monthc}     # orizontal 0 reflectance 

g.mremove -f  rast=cloud${day}_${tile}_coef,aeros${day}_${tile}_coef

r.out.gdal -c type=Float32  nodata=-1  createopt="COMPRESS=LZW,ZLEVEL=9"  input=diff_HradCA_day${day}_month${monthc}    output=$OUTDIR/diff_Hrad_day_tiles/$day/diff_HradCA_day$day"_"month$monthc"_"$tile.tif 
r.out.gdal -c type=Float32  nodata=-1  createopt="COMPRESS=LZW,ZLEVEL=9"  input=beam_HradCA_day${day}_month${monthc}    output=$OUTDIR/beam_Hrad_day_tiles/$day/beam_HradCA_day$day"_"month$monthc"_"$tile.tif 
# r.out.gdal -c type=Float32  nodata=-1  createopt="COMPRESS=LZW,ZLEVEL=9"  input=refl_HradCA_day${day}_month${monthc}  output=$OUTDIR/refl_Hrad_day_tiles/$day/refl_HradCA_day$day"_"month$monthc"_"$tile.tif 



' _

echo start  month $monthc average 

for INPUT  in T  CA; do 

r.series input=$(g.mlist rast pattern="diff_Hrad${INPUT}_day*_month${monthc}" sep=,)   output=tdiff_Hrad${INPUT}_m$monthc   method=average  --overwrite 
r.series input=$(g.mlist rast pattern="beam_Hrad${INPUT}_day*_month${monthc}" sep=,)   output=tbeam_Hrad${INPUT}_m$monthc   method=average  --overwrite 
# r.series input=$(g.mlist rast pattern="refl_Hrad${INPUT}_day*_month${monthc}" sep=,)   output=trefl_Hrad${INPUT}_m$monthc   method=average  --overwrite   # removed , always 0 in the horizontal situation 

r.mapcalc   "diff_Hrad${INPUT}_m$monthc  = float (  tdiff_Hrad${INPUT}_m$monthc )"
r.mapcalc   "beam_Hrad${INPUT}_m$monthc  = float (  tbeam_Hrad${INPUT}_m$monthc )"
# r.mapcalc   "refl_Hrad${INPUT}_m$monthc  = float (  trefl_Hrad${INPUT}_m$monthc )"   # removed , always 0 in the horizontal situation 

g.remove rast=tdiff_Hrad${INPUT}_m$monthc
g.remove rast=tbeam_Hrad${INPUT}_m$monthc
# g.remove rast=trefl_Hrad${INPUT}_m$monthc

r.out.gdal -c type=Float32  nodata=-1  createopt="COMPRESS=LZW,ZLEVEL=9"  input=diff_Hrad${INPUT}_m$monthc    output=$OUTDIR/diff_Hrad_month_tiles/$monthc/diff_Hrad${INPUT}_month$monthc"_"$tile.tif   --q --o
r.out.gdal -c type=Float32  nodata=-1  createopt="COMPRESS=LZW,ZLEVEL=9"  input=beam_Hrad${INPUT}_m$monthc    output=$OUTDIR/beam_Hrad_month_tiles/$monthc/beam_Hrad${INPUT}_month$monthc"_"$tile.tif  --q --o
# r.out.gdal -c type=Float32  nodata=-1  createopt="COMPRESS=LZW,ZLEVEL=9"  input=refl_Hrad${INPUT}_m$monthc    output=$OUTDIR/refl_Hrad_month_tiles/$monthc/refl_Hrad${INPUT}_month$monthc"_"$tile.tif   # removed , always 0 in the horizontal situation 

# remove the monthly mean 
g.mremove -f  rast=glob_Hrad${INPUT}_m$monthc,diff_Hrad${INPUT}_m$monthc,beam_Hrad${INPUT}_m$monthc,refl_Hrad${INPUT}_m$monthc


done 

g.mremove -f  rast=diff_HradT_day*_month${monthc},beam_HradT_day*_month${monthc},refl_HradT_day${day}_month${monthc}
g.mremove -f  rast=diff_HradCA_day*_month${monthc},beam_HradCA_day*_month${monthc},refl_HradCA_day${day}_month${monthc}

done 

rm -rf  /dev/shm/*

fi # close the first if condition 

checkjob -v $PBS_JOBID

exit 








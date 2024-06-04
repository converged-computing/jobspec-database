#!/bin/bash
#PBS -N huc12hand
#PBS -e /gpfs_scratch/nfie/huc12/stderr
#PBS -o /gpfs_scratch/nfie/huc12/stdout
#PBS -l nodes=10:ppn=20,walltime=32:00:00
#PBS -M yanliu@ncsa.illinois.edu
#PBS -m be

## handbyhuc.sh: create Height Above Nearest Drainage raster by HUC code.
## version: v0.11
## Author: Yan Y. Liu <yanliu.illinois.edu>
## Date: 05/27/2016
## This is a script to demonstrate all the steps needed to create HAND.

# env setup
module purge
module load MPICH gdal2-stack GCC/4.9.2-binutils-2.25
sdir=/projects/nfie/nfie-floodmap/test
source $sdir/handbyhuc.env

# config
hucid='120402'
n='gbay'
hucid='12090205'
n='travis'
[ ! -z "$1" ] && hucid="$1"
huclen=${#hucid}
[ ! -z "$2" ] && n="$2"
np="$3"
[ -z "$np" ] && np=$PBS_NP && [ -z "$np" ] && np=20

T00=`date +%s` 
echo "[`date`] Running HAND workflow for HUC$hucid($n) using $np cores..."
#wdir=/gpfs_scratch/nfie/${n}
wdir=`pwd`/${n}
cdir=`pwd`
mkdir -p $wdir
cd $wdir

echo "=1=: create watershed boundary shp from WBD"
echo -e "\tThis step queries WBD to get the boundary shp of study watershed."
echo "=1= ogr2ogr ${n}-wbd.shp $dswbd WBDHU${huclen} -where \"HUC${huclen}='${hucid}'\" "
Tstart
[ ! -f "${n}-wbd.shp" ] && \
ogr2ogr ${n}-wbd.shp $dswbd WBDHU${huclen} -where "HUC${huclen}='${hucid}'" \
&& [ $? -ne 0 ] && echo "ERROR creating watershed boundary shp." && exit 1
Tcount wbd

echo "=1.1=: buffer boundary shp to avoid edge contamination effect"
echo "ogr2ogr -dialect sqlite -sql \"select ST_buffer(Geometry, $bufferdist) from '${n}-wbd'\" ${n}-wbdbuf.shp ${n}-wbd.shp "
Tstart
[ ! -f "${n}-wbdbuf.shp" ] && \
ogr2ogr -dialect sqlite -sql "select ST_buffer(Geometry, $bufferdist) from '${n}-wbd'" ${n}-wbdbuf.shp ${n}-wbd.shp  \
&& [ $? -ne 0 ] && echo "ERROR buffering boundary shp." && exit 1
Tcount wbdbuf

echo "=2=: create DEM from NED 10m"
echo -e "\tThis step clips the DEM of the study watershed from the NED 10m VRT."
echo -e "\tThe output is hucid.tif of the original projection (geo)."
echo "gdalwarp -cutline ${n}-wbdbuf.shp -cl ${n}-wbdbuf -crop_to_cutline -of \"GTiff\" -overwrite -co \"COMPRESS=LZW\" -co \"BIGTIFF=YES\" $dsdem ${n}.tif "
Tstart
[ ! -f "${n}.tif" ] && \
gdalwarp -cutline ${n}-wbdbuf.shp -cl ${n}-wbdbuf -crop_to_cutline -of "GTiff" -overwrite -co "COMPRESS=LZW" -co "BIGTIFF=YES" $dsdem ${n}.tif \
&& [ $? -ne 0 ] && echo "ERROR clipping study area DEM." && exit 1
Tcount dem

echo "=3=: create flowline shp from NHDPlus"
echo "=3CMD= ogr2ogr ${n}-flows.shp $dsnhdplus Flowline -where \"REACHCODE like '${hucid}%'\""
Tstart
[ ! -f "${n}-flows.shp" ] && \
ogr2ogr ${n}-flows.shp $dsnhdplus Flowline -where "REACHCODE like '${hucid}%'" \
&& [ $? -ne 0 ] && echo "ERROR creating flowline shp." && exit 1
Tcount flowline

echo "=4=: find inlets from flowline shp"
find_inlets=$sdir/../src/find_inlets/build/find_inlets 
#find_inlets=/projects/nfie/hand/inlet-finder/build/find_inlets
echo "=4CMD= $find_inlets -flow ${n}-flows.shp -dangle ${n}-inlets0.shp "
Tstart
[ ! -f "${n}-inlets0.shp" ] && \
$find_inlets -flow ${n}-flows.shp -dangle ${n}-inlets0.shp \
&& [ $? -ne 0 ] && echo "ERROR creating inlet shp." && exit 1
Tcount dangle

echo "=5=: rasterize inlet points"
Tstart
[ ! -f "${n}-weights.tif" ] && \
ogr2ogr -t_srs $dsepsg ${n}-inlets.shp ${n}-inlets0.shp && \
read fsizeDEM colsDEM rowsDEM nodataDEM xmin ymin xmax ymax cellsize_resx cellsize_resy<<<$(python $sdir/getRasterInfo.py ${n}.tif) && \
echo "=5CMD= gdal_rasterize  -ot Int16 -of GTiff -burn 1 -tr $cellsize_resx $cellsize_resy -te $xmin $ymin $xmax $ymax ${n}-inlets.shp ${n}-weights.tif" && \
gdal_rasterize  -ot Int16 -of GTiff -burn 1 -tr $cellsize_resx $cellsize_resy -te $xmin $ymin $xmax $ymax ${n}-inlets.shp ${n}-weights.tif \
&& [ $? -ne 0 ] && echo "ERROR rasterizing inlet shp to weight grid." && exit 1
Tcount weights

#module purge
#module load MPICH gdal-stack

echo "=6=: taudem pitremove"
echo "=6CMD= mpirun -np $np $taudem/pitremove -z ${n}.tif -fel ${n}fel.tif"
Tstart
[ ! -f "${n}fel.tif" ] && \
mpirun -np $np $taudem/pitremove -z ${n}.tif -fel ${n}fel.tif \
&& [ $? -ne 0 ] && echo "ERROR creating pitremove DEM." && exit 1
Tcount pitremove 

echo "=7=: taudem dinf"
#echo "=7CMD= mpirun -np $np $taudem/dinfflowdir -fel ${n}fel.tif -ang ${n}ang.tif -slp ${n}slp.tif "
echo "=7CMD= mpirun -np $np $taudemdinf -fel ${n}fel.tif -ang ${n}ang.tif -slp ${n}slp.tif "
Tstart
[ ! -f "${n}ang.tif" ] && \
#mpirun -np $np $taudem/dinfflowdir -fel ${n}fel.tif -ang ${n}ang.tif -slp ${n}slp.tif \
mpirun -np $np $taudemdinf -fel ${n}fel.tif -ang ${n}ang.tif -slp ${n}slp.tif \
&& [ $? -ne 0 ] && echo "ERROR creating dinf raster." && exit 1
Tcount dinf

echo "=8=: taudem d8"
#echo "=8CMD= mpirun -np $np $taudem/d8flowdir -fel ${n}fel.tif -p ${n}p.tif -sd8 ${n}sd8.tif "
echo "=8CMD= mpirun -np $np $taudemd8 -fel ${n}fel.tif -p ${n}p.tif -sd8 ${n}sd8.tif "
Tstart
[ ! -f "${n}p.tif" ] && \
#mpirun -np $np $taudem/d8flowdir -fel ${n}fel.tif -p ${n}p.tif -sd8 ${n}sd8.tif \
mpirun -np $np $taudemd8 -fel ${n}fel.tif -p ${n}p.tif -sd8 ${n}sd8.tif \
&& [ $? -ne 0 ] && echo "ERROR creating d8 raster." && exit 1
Tcount d8

echo "=9=: taudem aread8 with weights"
echo "=9CMD= mpirun -np $np $taudem/aread8 -p ${n}p.tif -ad8 ${n}ssa.tif -wg ${n}-weights.tif -nc"
Tstart
[ ! -f "${n}ssa.tif" ] && \
mpirun -np $np $taudem/aread8 -p ${n}p.tif -ad8 ${n}ssa.tif -wg ${n}-weights.tif -nc \
&& [ $? -ne 0 ] && echo "ERROR creating aread8 raster with weights." && exit 1
Tcount aread8w

echo "=10=: taudem threshold"
echo "=10CMD= mpirun -np $np $taudem/threshold -ssa ${n}ssa.tif -src ${n}src.tif -thresh 1 "
Tstart
[ ! -f "${n}src.tif" ] && \
mpirun -np $np $taudem/threshold -ssa ${n}ssa.tif -src ${n}src.tif -thresh 1 \
&& [ $? -ne 0 ] && echo "ERROR creating streamgrid using threshold." && exit 1
Tcount threshold

echo "=11=: taudem dinfdistdown"
echo "=11CMD= mpirun -np $np $taudem/dinfdistdown -fel ${n}fel.tif -ang ${n}ang.tif -src ${n}src.tif -dd ${n}dd.tif -m ave v"
Tstart
[ ! -f "${n}dd.tif" ] && \
mpirun -np $np $taudem/dinfdistdown -fel ${n}fel.tif -ang ${n}ang.tif -src ${n}src.tif -dd ${n}dd.tif -m ave v \
&& [ $? -ne 0 ] && echo "ERROR creating HAND raster." && exit 1
Tcount dinfdistdown

echo "=12=: clip DistDown raster to original WBD size"
echo "gdalwarp does a bad job in clipping, which results in huge output raster."
echo "so we create an uncompressed tmp hand; then move it to the right size."
echo "gdalwarp -cutline ${n}-wbd.shp -cl ${n}-wbd -crop_to_cutline -of "GTiff" -overwrite -co "COMPRESS=LZW" -co "BIGTIFF=YES" ${n}dd.tif ${n}hand.tif "
Tstart
[ ! -f "${n}hand.tif" ] && \
gdalwarp -cutline ${n}-wbd.shp -cl ${n}-wbd -crop_to_cutline -of "GTiff" -overwrite -co "BIGTIFF=YES" ${n}dd.tif ${n}handtmp.tif \
&& gdal_translate -of GTiff -co "COMPRESS=LZW" -co "BIGTIFF=YES" ${n}handtmp.tif ${n}hand.tif \
&& rm -f ${n}handtmp.tif \
&& [ $? -ne 0 ] && echo "ERROR clipping DistDown raster to original WBD boundary" && exit 1
Tcount hand

T01=`date +%s`
echo "=STAT= $hucid $T00 `expr $T01 \- $T00` $np" 
cd $cdir
echo "[`date`] Finished."

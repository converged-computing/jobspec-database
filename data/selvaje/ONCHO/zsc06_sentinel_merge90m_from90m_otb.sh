#!/bin/bash
#SBATCH -p scavenge
#SBATCH -n 1 -c 1 -N 1
#SBATCH -t 24:00:00 
#SBATCH -o /vast/palmer/scratch/sbsc/ga254/stdout/zsc06_sentinel_merge90m_from90m_otb.sh.%A_%a.out
#SBATCH -e /vast/palmer/scratch/sbsc/ga254/stderr/zsc06_sentinel_merge90m_from90m_otb.sh.%A_%a.err
#SBATCH --job-name=zsc06_sentinel_merge90m_from90m_otb.sh
#SBATCH --mem=120G
#SBATCH --array=1-4

#### sbatch /vast/palmer/home.grace/ga254/scripts_gitreps/ONCHO/zsc06_sentinel_merge90m_from90m_otb.sh

source ~/bin/gdal3

ONCHO=/gpfs/gibbs/project/sbsc/ga254/dataproces/ONCHO
SENT=/gpfs/gibbs/project/sbsc/ga254/dataproces/ONCHO/input_orig/sentinel
IN=/gpfs/gibbs/project/sbsc/ga254/dataproces/ONCHO/input_orig/sentinel/tif_wgs84_90m_tile
OUT=/gpfs/gibbs/project/sbsc/ga254/dataproces/ONCHO/input_orig/sentinel/tif_wgs84_90m_from90m_otb
RAM=/dev/shm
                                                   # xmin ymin xmax ymax
                                               #       2   4    15    15 
#  gdalbuildvrt -srcnodata 0 -vrtnodata 0 -overwrite -te  2  10     8 15 $IN/NW.vrt $IN/*.tif 
#  gdalbuildvrt -srcnodata 0 -vrtnodata 0 -overwrite -te  8  10    15 15 $IN/NE.vrt $IN/*.tif 
#  gdalbuildvrt -srcnodata 0 -vrtnodata 0 -overwrite -te  2  4      8 10 $IN/SW.vrt $IN/*.tif 
#  gdalbuildvrt -srcnodata 0 -vrtnodata 0 -overwrite -te  8  4     15 10 $IN/SE.vrt $IN/*.tif 

export file=$(ls $IN/??.vrt   | head -$SLURM_ARRAY_TASK_ID | tail -1 )
export filename=$(basename $file .vrt  )

##### oft 
##### cd ~/bin
##### apptainer build  orfeotoolbox.sif   docker://orfeotoolbox/otb

apptainer exec  ~/bin/orfeotoolbox.sif bash <<EOF
export GDAL_CACHEMAX=99999
export OTB_MAX_RAM_HINT=10000

# The slim blending composition mode  - Blends the last image over earlier ones in areas of overlap, on a given transition distance (seam can be visible from a certain zoom level, but it does not cause blur effect when images are not perfectly aligned) 

otbcli_Mosaic -interpolator linear  -output.spacingx 0.0008333333333333  -output.spacingy 0.0008333333333333 -nodata 0 -harmo.method band -comp.feather slim -comp.feather.slim.length  0.008333333333333 -il $(gdalinfo $file | grep "\.tif" | awk '{ printf ("%s " , $1 ) }' )  -out $OUT/tile_all_90m_otb_${filename}4.tif  -ram 10000
gdal_translate -ot  UInt16 -projwin $(getCorners4Gtranslate $file)  -a_ullr $(getCorners4Gtranslate $file)  -co COMPRESS=DEFLATE -co ZLEVEL=9 $OUT/tile_all_90m_otb_${filename}4.tif $OUT/tile_all_90m_otb_slim_harm_${filename}.tif
rm $OUT/tile_all_90m_otb_${filename}4.tif

# The large blending composition mode - Blends all images on the maximum overlapping areas (this produces seamless mosaics, but can cause blur effect where images are not perfectly aligned)

otbcli_Mosaic -interpolator linear  -output.spacingx 0.0008333333333333  -output.spacingy 0.0008333333333333 -nodata 0  -comp.feather large -il $(gdalinfo $file | grep "\.tif" | awk '{ printf ("%s " , $1 ) }' )  -out $OUT/tile_all_90m_otb_${filename}1.tif  -ram 10000
gdal_translate -ot  UInt16 -projwin $(getCorners4Gtranslate $file)  -a_ullr $(getCorners4Gtranslate $file)  -co COMPRESS=DEFLATE -co ZLEVEL=9 $OUT/tile_all_90m_otb_${filename}1.tif $OUT/tile_all_90m_otb_large_${filename}.tif
rm $OUT/tile_all_90m_otb_${filename}1.tif

otbcli_Mosaic -interpolator linear  -output.spacingx 0.0008333333333333  -output.spacingy 0.0008333333333333 -nodata 0  -comp.feather large -harmo.method band   -il $(gdalinfo $file | grep "\.tif" | awk '{ printf ("%s " , $1 ) }'  )  -out $OUT/tile_all_90m_otb_${filename}2.tif  -ram 10000
gdal_translate -ot  UInt16 -projwin $(getCorners4Gtranslate $file)  -a_ullr $(getCorners4Gtranslate $file)  -co COMPRESS=DEFLATE -co ZLEVEL=9 $OUT/tile_all_90m_otb_${filename}2.tif $OUT/tile_all_90m_otb_large_harm_${filename}.tif
rm $OUT/tile_all_90m_otb_${filename}2.tif

otbcli_Mosaic -interpolator linear  -output.spacingx 0.0008333333333333  -output.spacingy 0.0008333333333333 -nodata 0  -comp.feather slim -comp.feather.slim.length 0.008333333333333 -il $(gdalinfo $file | grep "\.tif" | awk '{ printf ("%s " , $1 ) }' )  -out $OUT/tile_all_90m_otb_${filename}3.tif  -ram 10000
gdal_translate -ot  UInt16 -projwin $(getCorners4Gtranslate $file)  -a_ullr $(getCorners4Gtranslate $file)  -co COMPRESS=DEFLATE -co ZLEVEL=9 $OUT/tile_all_90m_otb_${filename}3.tif $OUT/tile_all_90m_otb_slim_${filename}.tif
rm $OUT/tile_all_90m_otb_${filename}3.tif


EOF







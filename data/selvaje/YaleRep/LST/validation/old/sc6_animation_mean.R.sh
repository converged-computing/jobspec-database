

# for  SENS in MYD MOD ; do for DN in Day Nig ; do qsub -v SENS=$SENS,DN=$DN   /u/gamatull/scripts/LST/validation/sc6_animation_mean.R.sh ; done ; done 


#PBS -S /bin/bash
#PBS -q normal
#PBS -l select=1:ncpus=4
#PBS -l walltime=8:00:00
#PBS -V
#PBS -o /nobackup/gamatull/stdout
#PBS -e /nobackup/gamatull/stderr

module load R/3.1.1_rgal_nex 

export SENS=$SENS
export DN=$DN

R --vanilla --no-readline   -q  <<'EOF'

SENS = Sys.getenv(c('SENS'))
DN = Sys.getenv(c('DN'))


# source ("/u/gamatull/scripts/LST/validation/sc6_animation_mean.R.sh")
.libPaths( c( .libPaths(), "/home/fas/sbsc/ga254/R/x86_64-unknown-linux-gnu-library/3.0") )

library(raster)
library(lattice)
library(rasterVis)
library(foreach)

# gdal_translate -co COMPRESS=LZW -co ZLEVEL=9 -ot Int32 -a_nodata -9999 integration001.tif integration001a.tif
rmr=function(x){
## function to truly delete raster and temporary files associated with them
if(class(x)=="RasterLayer"&grepl("^/tmp",x@file@name)&fromDisk(x)==T){
file.remove(x@file@name,sub("grd","gri",x@file@name))
rm(x)
}
}

path = paste0("/nobackupp8/gamatull/dataproces/LST/",SENS,"11A2_mean/wgs84/")

# for parallel
# you can use a foreach to do it in parallel too, if you like
# foreach(file=files.list) %dopar%{

day.list=read.table("/nobackupp8/gamatull/dataproces/LST/geo_file/list_day.txt" , colClasses=c("character") )

for (i in 1:46)  {

day=day.list[i,]

basename=paste("LST_",SENS,"_QC_day",day,"_wgs84k10_",DN,sep="")  
## not sure exactly what this does, but there is basename(file) too...
png(paste(path,basename,".png",sep=""),width=2000,height=1000)
## add width and height to specify the size of the png, otherwise it will be small (100x100, I think)

day001=((raster(paste(path,basename,".tif", sep="")) * 0.02) - 272.15 )

## be careful with this, it will recalculate the whole image into a temporary directory and not automatically delete it..
## I would suggest deleting it at the end of the loop...
n=100
at=seq(-50,90,length=n)
colR=colorRampPalette(c("blue","green","yellow", "red", "brown"))
cols=colR(n)
res=1e4  # res=1e4 for testing and res=1e6 for the final product
greg=list(ylim=c(-90,90),xlim=c(-180,180))

print ( levelplot(day001,col.regions=colR(n) , cuts=99,at=at,colorkey=list(space="right",adj=1), panel=panel.levelplot.raster,margin=F,maxpixels=res,ylab="",xlab="",useRaster=T,ylim=greg$ylim) + layer(panel.text(160, 85, paste('Julian day ',day,sep=""),cex=2 )) )
rmr(day001) # really remove raster files, this will delete the temporary file
gc()
dev.off()
}

# move the data to litoria /home/giuseppea/tmp/animation_MERRAero and run ffmepg
# ffmpeg -r 25 -i mean%03d.png -vcodec libx264 -pix_fmt yuv420p -r 25 animation_MERRAero.mp4
q()
EOF
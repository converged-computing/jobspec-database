#!/bin/sh
#PBS -S /bin/sh
#PBS -N PSFsel_@FILTER@
#PBS -o PSFsel.out
#PBS -j oe
#PBS -l nodes=1:ppn=2,walltime=04:00:00
#-----------------------------------------------------------------------------
# PSFsel: find frames with bad psf and discard
# requires: intelpython, astropy.io.fits, uvis scripts and libs
#-----------------------------------------------------------------------------

set -u  
umask 022
ec()  { echo    "$(date "+[%d.%h.%y %T"]) $1 " ; }
ecn() { echo -n "$(date "+[%d.%h.%y %T"]) $1 " ; }
mycd() { \cd $1; ec " --> $PWD"; }               # cd with message

module() {  eval $(/usr/bin/modulecmd bash $*); }
module purge ; module load intelpython/2

#-----------------------------------------------------------------------------
# Some variables
#-----------------------------------------------------------------------------

sdate=$(date "+%s")

uvis=/home/moneti/softs/uvis-pipe      # top UltraVista code dir
bindir=$uvis/bin
pydir=$uvis/python
confdir=$uvis/config                   # location of support and config files
export PYTHONPATH=$pydir

#-----------------------------------------------------------------------------

module=PSFsel                          # w/o .sh extension

# check  if run via shell or via qsub:
if [[ "$0" =~ "$module" ]]; then
    ec "$module: running as shell script "
    WRK=$WRK
    FILTER=$FILTER
	list=$1
    if [[ "${@: -1}" =~ 'dry' ]] || [ "${@: -1}" == 'test' ]; then dry=T; else dry=F; fi
else
    ec "$module: running via qsub (from pipeline)"
    dry=0
    WRK=@WRK@
	list=@LIST@
    FILTER=@FILTER@
fi

badfiles=$WRK/DiscardedFiles.list      # built and appended to during processing

#-----------------------------------------------------------------------------
mycd $WRK/images
if [ $? -ne 0 ]; then echo "ERROR: $WRK/images not found ... quitting"; exit 5; fi
echo "------------------------------------------------------------------"

sdate=$(date "+%s")

ec "# extract seeing and ellipticity data from qFits psfex.xml files"
mexec=" $pydir/psfxml2dat.py " 

# fix a minor problem in the xml files
sed -i 's|pted_Mean\"\ datatype=\"int|pted_Mean\"\ datatype=\"float|' v20*.xml

outfile=PSFsel.dat ; rm -f $outfile         # just in case
nn=$(cat PSFsel.lst | wc -l)
ec "# Found $nn _psfex.xml to work on ..."

comm="python $mexec -l $list -o $outfile "
ec "# command line is:"
ec "% $comm"

if [ $dry == 'T' ]; then echo "   >> EXITING TEST MODE << "; exit 0; fi

$comm
if [ $? -ne 0 ]; then
    ec "## PROBLEM ## $module.sh - walltime: $dt sec "
    tail $mlogfile 
    ec "------------------------------------------------------------------"
    exit 1
fi

if [ ! -s $outfile ]; then echo "PROBLEM: psfstats file emtpy ... "; fi

nerr=$(grep Error $outfile | wc -l)
if [ $nerr -gt 0 ]; then ec "# Attn: found $nerr missing psfex.xml files; continue with others "; fi

# ---------------------- find images with large psf ----------------------

ec "# Look for images with too large (> 1 arcsec) or too elliptical (> 0.1) PSF ..."
#echo '#   file          FWHM (")  Ell' > badPSF.dat
awk '/v20/{if ($2 > 1.0 || $3 > 0.1) printf "%-18s %6.3f  %6.4f\n", $1, $2, $3 }' $outfile >> badPSF.dat
nbad=$(grep v20 badPSF.dat | wc -l) 

if [ $nbad -gt 0 ]; then 
    ec "# Found $nbad files with bad psf - see images/badPSF.dat " 
    # ---------------------- discard files from badPSF.dat ----------------------
	# for DR5: do not reject these files: keep them for making sky
else
    ec "# ... none found"  
#    rm badPSF.dat
#    echo " No bad PSF files found ... " >> badPSF.dat  # to have a non-empty file
fi

#-----------------------------------------------------------------------------
edate=$(date "+%s"); dt=$(($edate - $sdate))
echo " >>>> $module.sh finished - walltime: $dt sec  <<<<"
echo "------------------------------------------------------------------"
echo ""
exit 0

#-----------------------------------------------------------------------------

#!/bin/bash 
#PBS -S /bin/sh
#PBS -N pscamp_@PTAG@_@FILTER@
#PBS -o @IDENT@.out
#PBS -j oe
#PBS -l nodes=1:ppn=9:hasnogpu,walltime=@WTIME@:00:00
#-----------------------------------------------------------------------------
# pscamp: run scamp on a list of ldacs
# requires: astromatic suite, ... intelpython, astropy.io.fits, uvis scripts and libs
#-----------------------------------------------------------------------------
set -u 
export PATH="/softs/astromatic/bin:$PATH"  #echo $PATH
export PYTHONPATH="/home/moneti/uvis/python:/home/moneti/uvis/python_lib" 

#-----------------------------------------------------------------------------
# this is for Henry's scamp with 
#-----------------------------------------------------------------------------
module() { eval $(/usr/bin/modulecmd bash $*); }
module purge ; module load inteloneapi/2021.1 intelpython/3-2019.4 cfitsio
export LD_LIBRARY_PATH=/lib64:${LD_LIBRARY_PATH}

#-----------------------------------------------------------------------------
# other functions
#-----------------------------------------------------------------------------

ec() { echo "$(date "+[%d.%h.%y %T"]) $1 "; }    # echo with date
ec() { echo "[pscamp.sh]" $1; }    # echo with scamp
dt() { echo "$(date "+%s.%N") $bdate" | awk '{printf "%0.2f\n", $1-$2}'; }
wt() { echo "$(date "+%s.%N") $bdate" | awk '{printf "%0.2f hrs\n", ($1-$2)/3600}'; }  # wall time

#-----------------------------------------------------------------------------
# Some variables
#-----------------------------------------------------------------------------

module=pscamp_@PTAG@               # w/o .sh extension
uvis=/home/moneti/softs/uvis-pipe # top UltraVista code dir
bindir=$uvis/bin
pydir=$uvis/python                # python scripts
scripts=$uvis/scripts             # other scripts dir (awk ...)
export confdir=$uvis/config              # config dir

# check  if run via shell or via qsub:
ec "#-----------------------------------------------------------------------------"
if [[ "$0" =~ "$module" ]]; then
    ec "# $module: running as shell script on $(hostname)"
	list=@LIST@
	dry=@DRY@
	WRK=@WRK@
	ptag=@PTAG@
	pass=@PASS@
	FILTER=$FILTER
	verb=" -VERBOSE_TYPE LOG"
	pipemode=0
else
    ec "# $module: running via qsub (from pipeline) on $(hostname)"
	WRK=@WRK@
	dry=@DRY@
	ptag=@PTAG@
	list=@LIST@
	FILTER=@FILTER@
	pass=@PASS@
	verb=" -VERBOSE_TYPE LOG" # QUIET"
	pipemode=1
fi
export FILTER=$FILTER

ec "#-----------------------------------------------------------------------------"

#-----------------------------------------------------------------------------------------------
case  $FILTER in 
   N | NB118) magzero=29.14 ;;   # FILTER=NB118 ;;
   Y | Q    ) magzero=29.39 ;;
   J | R    ) magzero=29.10 ;;
   H | S    ) magzero=28.62 ;;
   K | Ks   ) magzero=28.16 ;;   # FILTER=Ks    ;;
   * ) ec "# ERROR: invalid filter $FILTER"; exit 3 ;;
esac   
#-----------------------------------------------------------------------------------------------

cd $WRK

nldacs=$(cat $list | wc -l)
for f in $(cut -c1-15  $list); do ls ${f}.ahead 2> /dev/null ; done > ${list}.aheads
naheads=$(cat ${list}.aheads 2> /dev/null | wc -l) ; rm ${list}.aheads
if [ $naheads -eq 0 ]; then
	ec "# PROBLEM: no photref files for this list"
fi

ec "# Using $list with $nldacs files and $naheads photref  files "
ec "# Filter is $FILTER; magzero = $magzero" 
ec "# Using scamp  ==> $(scamp -v)"
sconf=$confdir/scamp_dr6.conf   # new one for DR5
ec "# scamp config file is $sconf"

logfile=$WRK/$module.log ; rm -f $logfile
season=$(echo $list | cut -c9-10)
args=" -c $sconf  -MAGZERO_OUT $magzero  -ASTRINSTRU_KEY OBJECT "

if [[ ${pass: -1} =~ 'm' ]]; then
	ec "# ###  Using season-specific GAIA reference catalgues   ###"
	catal="-ASTREFCAT_NAME $confdir/GAIA-EDR3_s${season}.cat"       # multi: Catals by season
else
	ec "# ###  Using single GAIA reference catalgues for full survey ###"
	catal="-ASTREFCAT_NAME $confdir/GAIA-EDR3_1000+0211_r61.cat"   # single: GAIA catal for all times
fi

ahead="-AHEADER_GLOBAL ${confdir}/vista_${FILTER}.ahead  -MOSAIC_TYPE SAME_CRVAL"   # in config file
extra="-ASTRACCURACY_TYPE TURBULENCE-ARCSEC -ASTR_ACCURACY 0.054  -POSITION_MAXERR 2.8  -XML_NAME $module.xml"
ptype="-CHECKPLOT_TYPE  FGROUPS,ASTR_REFERROR1D,ASTR_REFERROR2D"
pname="-CHECKPLOT_NAME groups_${ptag},refe1d_${ptag},refe2d_${ptag}"

# build command line
comm="scamp @$list  $args  $ahead  $catal  $extra $ptype $pname $verb"

ec "# PBS resources: $(head $WRK/$module.sh | grep nodes= | cut -d \  -f3)"
ec "# logfile is $logfile"
ec "# Command line is:"
ec "    $comm"
ec ""
if [[ $dry == 'T' ]]; then
	echo "[---DRY---] Working directory is $WRK"
	echo "[---DRY---] Input files are like $(tail -1 $list)"
    echo "[---DRY---] >>  Dry-run of $0 finished .... << "
	ec "#-----------------------------------------------------------------------------"
#	for f in $(cat $list); do rm $f .; done
	exit 0
else
	for f in $(cat $list); do ln -sf ldacs/$f .; done
fi

#-----------------------------------------------------------------------------

bdate=$(date "+%s.%N")

if [ $pipemode -eq 0 ]; then
	ec " shell mode" > $logfile
	if [ $nldacs -lt 125 ]; then 
		$comm  2>&1  | tee -a $logfile   # output also to screen if few files
	else 
		$comm >> $logfile 2>&1           # otherwise to logfile only
	fi
	if [ $? -ne 0 ]; then ec "Problem ... " ; exit 5; fi
else    # qsub mode
	ec "  $comm"  >> $logfile
	ec " "    >> $logfile
	$comm     >> $logfile 2>&1     # output to logfile in pipeline mode
	if [ $? -ne 0 ]; then ec "Problem ... " ; tail $logfile ; exit 5; fi
fi

#-----------------------------------------------------------------------------
nerr=$(grep Error $logfile | wc -l);
if [ $nerr -ge 1 ]; then
    grep Error $logfile
    ec "# PROBLEM: $nerr errors found in $logfile ... quitting"
	exit 5 
fi

# check for warnings ==> pscamp.warn"
grep WARNING $logfile | grep -v -e FLAGS\ param -e ATLAS > $WRK/$module.warn
nw=$(cat $WRK/$module.warn | wc -l )
if [ $nw -ne 0 ]; then 
	ec "#### ATTN: $nw warnings found !!"; 
else 
	ec "# ... NO warnings found - congrats"; rm $WRK/$module.warn
fi 

# extract table 3 form xml file
$pydir/scamp_xml2dat.py $module.xml 

# rename the pngs to have the filter name and the pass - just to rename the png files
#if [ $FILTER == 'NB118' ]; then FILTER='N'; fi
#if [ $FILTER == 'Ks' ];    then FILTER='K'; fi

rename _1.png _${FILTER}.png [g,i,r,p]*_1.png

#-----------------------------------------------------------------------------
# and finish up
#-----------------------------------------------------------------------------
ec " >>>>  pscamp finished - walltime: $(wt)  <<<<"
ec "#-----------------------------------------------------------------------------"
ec ""

for f in $(cat $list); do rm $f ; done   # cleanup
exit 0

#-----------------------------------------------------------------------------

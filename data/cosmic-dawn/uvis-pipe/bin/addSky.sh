#!/bin/sh
#PBS -S /bin/sh
#PBS -N addSky_@FILTER@_@ID@
#PBS -o @IDENT@.out            
#PBS -j oe
#PBS -l nodes=1:ppn=3,walltime=24:00:00
#-----------------------------------------------------------------------------
# template:  addSky script
# requires: intelpython, astropy.io.fits, uvis scripts and libs
#-----------------------------------------------------------------------------

set -u  
export PYTHONPATH="/home/moneti/uvis/python:/home/moneti/uvis/python_lib"

# add python and UltraVista scripts and libs
module () {  eval $(/usr/bin/modulecmd bash $*); }
module purge; module load intelpython/2

#-----------------------------------------------------------------------------
# Some variables and functions
#-----------------------------------------------------------------------------

ec() { echo "$(date "+[%d.%h.%y %T"]) $1 "; }    # echo with date
sdate=$(date "+%s")

uvis=/home/moneti/softs/uvis-pipe    # top UltraVista code dir
bindir=$uvis/bin                     # pipeline modules
pydir=$uvis/python                   # python scripts
confdir=$uvis/config                 # config dir
errcode=0

#-----------------------------------------------------------------------------
# Setup
#-----------------------------------------------------------------------------
module=addSky                   # w/o .sh extension

# check  if run via shell or via qsub: 
if [[ "$0" =~ "$module" ]]; then
    echo "$module: running as shell script "
	list=$1
	WRK=$WRK
	FILTER=$FILTER
	if [[ "${@: -1}" =~ 'dry' ]]; then dry=T; else dry=F; fi
	logfile=addSky_$list.log
else
    echo "$module: running via qsub (from pipeline)"
	dry=@DRY@
	list=@LIST@
	FILTER=@FILTER@
	WRK=@WRK@
	logfile=addSky_@ID@.log
fi

#-----------------------------------------------------------------------------
# The REAL work ... done locally in images dir
#-----------------------------------------------------------------------------

cd $WRK/images

if [ $? -ne 0 ]; then echo "ERROR: $WRK/images not found ... quitting"; exit 5; fi
if [ ! -s $list ]; then echo "ERROR: $list not found in $WRK/images ... quitting"; exit 5; fi

nl=$(cat $list | wc -l)
echo "=================================================================="
ec " >>>>  Begin addSky on $list with $nl entries  <<<<"
echo "------------------------------------------------------------------"

bdate=$(date "+%s")
comm="python $pydir/addSky.py -t $list -o _withSky.fits"
echo "% $comm  "
if [ $dry == 'T' ]; then
    echo " ## DRY MODE - do nothing ## "
	exit 0
else
	$comm 
fi

#-----------------------------------------------------------------------------
# and finish up
#-----------------------------------------------------------------------------
edate=$(date "+%s"); dt=$(($edate - $sdate))
echo " >>>> $module.sh finished - walltime: $dt sec  <<<<"
echo "------------------------------------------------------------------"
echo ""
exit $errcode

#-----------------------------------------------------------------------------
 

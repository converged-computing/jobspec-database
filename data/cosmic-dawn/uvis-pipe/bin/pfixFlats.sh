#!/bin/sh
#PBS -S /bin/sh
#PBS -N pflats_@FILTER@
#PBS -o pflats.out            
#PBS -j oe
#PBS -l nodes=@NODE@:ppn=4,walltime=24:00:00
#-----------------------------------------------------------------------------
# module: pfixflats.sh 
# requires: intelpython2, astropy.io.fits, uvis scripts and libs
#-----------------------------------------------------------------------------
set -u  

ec() { echo "$(date "+[%d.%h.%y %T"]) $1 "; }    # echo with date
mycd() { \cd $1; echo " --> $PWD"; }               # cd with message
sdate=$(date "+%s")

uvis=/home/moneti/softs/uvis-pipe   #_dr4
bindir=$uvis/bin
pydir=$uvis/python
confdir=$uvis/config

echo "# Using to python 2"
module() { eval $(/usr/bin/modulecmd bash $*); }
module purge; module load cfitsio intelpython/2-2019.4

export PATH="${uvis}/bin:$PATH"
export PYTHONPATH="${uvis}/python:${uvis}/python_lib"

#-----------------------------------------------------------------------------
# Setup
#-----------------------------------------------------------------------------
module=pfixFlats                     # w/o .sh extension

# check  if run via shell or via qsub: 
if [[ "$0" =~ "$module" ]]; then
    echo "# $module: running as shell script "
    list=list_flats   #$1
    WRK=$WRK
    FILTER=$FILTER
    if [[ "${@: -1}" =~ 'dry' ]] || [[ "${@: -1}" == 'test' ]]; then dry=T; else dry=F; fi
else
    echo "# $module: running via qsub (from pipeline)"
    dry=@DRY@
    list=@LIST@
    FILTER=@FILTER@
    WRK=@WRK@
fi

#-----------------------------------------------------------------------------
# The real work ....
#-----------------------------------------------------------------------------

#echo $PATH
mycd $WRK/calib

echo "#-----------------------------------------------------------------------------"
echo "## - remove PV from the headers and normalise" 
echo "#-----------------------------------------------------------------------------"
 
echo "# --- Begin python fix flats ---"
echo "#"

mexec="$pydir/convert_flats.py"
margs=" --verbosity=INFO --log=clean_flats.log"
mlogfile=fix_flats_out.log

comm="python $mexec -l $list $margs"
nims=$(cat $list | wc -l)

echo "# Command line, run in $(pwd), is:"
echo "# $comm"

if [[ $dry != "T" ]]; then
	$comm > $mlogfile 2>&1 
	if [ $? -ne 0 ]; then
		ec "## PROBLEM ## $module.sh - "
		tail $mlogfile 
		ec "------------------------------------------------------------------"
		exit 1
	else
		rm $mlogfile
	fi
else
	echo "#  ---- DRY MODE ... do nothing ---- "
fi

echo "#-----------------------------------------------------------------------------"
echo "# --- Begin python normalize flats ---"
echo "#"

mexec="$pydir/norm_flats.py"
margs=" --log=norm_flats.log"
mlogfile=norm_flats_out.log

comm="python $mexec -l $list $margs"

echo "# Command line, run in $(pwd), is:"
echo "# $comm"

if [[ $dry != "T" ]]; then
	$comm > $mlogfile 2>&1 
	if [ $? -ne 0 ]; then
		ec "## PROBLEM ## $module.sh  "
		tail $mlogfile 
		ec "------------------------------------------------------------------"
		exit 1
	else
		rm $mlogfile
	fi
else
	echo "#  ---- DRY MODE ... do nothing ---- "
fi

echo "#-----------------------------------------------------------------------------"



#-----------------------------------------------------------------------------
# and finish up
#-----------------------------------------------------------------------------
edate=$(date "+%s"); dt=$(($edate - $sdate))
echo " >>>> $module.sh finished - walltime: $dt sec  <<<<"
echo "------------------------------------------------------------------"
exit 0

#-----------------------------------------------------------------------------

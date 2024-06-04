#!/bin/sh
#PBS -S /bin/sh
#PBS -N pconv_@FILTER@@ID@
#PBS -o pconv_@ID@.out            
#PBS -j oe
#PBS -l nodes=@NODE@:ppn=4,walltime=24:00:00
#-----------------------------------------------------------------------------
# module: pconvert.sh 
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
module=pconvert                     # w/o .sh extension

# check  if run via shell or via qsub: 
if [[ "$0" =~ "$module" ]]; then
    echo "# $module: running as shell script "
    list=$1
    WRK=$WRK
    FILTER=$FILTER
    if [ $# -eq 2 ]; then dry=1; else dry=0; fi
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
mycd $WRK/images

mexec="$pydir/convert_images.py"
margs=" -s ../stacks/ --stack_addzp=Convert_keys.list --log=conv_wircam_@ID@.log"
mlogfile=convert_full_@ID@.log
comm="python $mexec -l $list $margs"
nims=$(cat $list | wc -l)

echo "# --- Begin conversion to WIRCam ---"

$comm > $mlogfile 2>&1 
if [ $? -ne 0 ]; then
	ec "## PROBLEM ## $module.sh - walltime: $dt sec "
	tail $mlogfile 
	ec "------------------------------------------------------------------"
	exit 1
 fi

#-----------------------------------------------------------------------------
# and finish up
#-----------------------------------------------------------------------------
edate=$(date "+%s"); dt=$(($edate - $sdate))
echo " >>>> $module.sh finished - walltime: $dt sec  <<<<"
echo "------------------------------------------------------------------"
exit 0

#-----------------------------------------------------------------------------

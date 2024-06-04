#!/bin/sh
#PBS -S /bin/sh
#PBS -N merge_@FILTER@_@TAG@
#PBS -o @IDENT@.out
#PBS -j oe 
#PBS -l nodes=1:ppn=6,walltime=08:00:00
#-----------------------------------------------------------------------------
# pmerge:  pmerge script - to merge substacks
# requires: intelpython, astropy.io.fits, uvis scripts and libs
#-----------------------------------------------------------------------------

set -u  
# paths
export PATH="~/bin:$PATH:/softs/dfits/bin:/softs/astromatic/bin"
export PYTHONPATH="/home/moneti/uvis/python:/home/moneti/uvis/python_lib"

# add python and UltraVista scripts and libs
module () {  eval $(/usr/bin/modulecmd bash $*); }
module purge; module load intelpython/2

#-----------------------------------------------------------------------------
# Some variables and functions
#-----------------------------------------------------------------------------

ec() { echo "$(date "+[%d.%h.%y %T"]) $1 "; }    # echo with date
ec() { if [ $dry == 'T' ]; then echo "[TEST MODE] $1";
    else echo "$(date "+[%h.%d %T]") $1 "; fi; } 
sdate=$(date "+%s")

module=pmerge                   # w/o .sh extension
uvis=/home/moneti/softs/uvis-pipe            # top UltraVista code dir
bindir=$uvis/bin
pydir=$uvis/python                # python scripts
confdir=$uvis/config              # config dir
errcode=0

#-----------------------------------------------------------------------------

# check  if run via shell or via qsub: 
if [[ "$0" =~ "$module" ]]; then
    echo "$module: running as shell script "
    list=$1
    WRK=$WRK
    FILTER=$FILTER
    if [[ "${@: -1}" == 'dry' ]]; then dry='T'; else dry='F'; fi
    stout=@STOUT@
    pass=2
else
    echo "$module: running via qsub (from pipeline)"
    dry=@DRY@
    list=@LIST@
    FILTER=@FILTER@
    WRK=@WRK@
    stout=@STOUT@
    pass=@PASS@
fi

verb=" -VERBOSE_TYPE LOG"
tail=$(echo ${list%.lst} | cut -d\_ -f2)

#-----------------------------------------------------------------------------
# The REAL work ... done locally in images dir
#-----------------------------------------------------------------------------

bdate=$(date "+%s")
cd $WRK/images

if [ $? -ne 0 ]; then echo "ERROR: $WRK/images not found ... quitting"; exit 5; fi
if [ ! -s $list ]; then echo "ERROR: $list not found in $WRK/images ... quitting"; exit 5; fi

# output file names:
stout=$stout.fits
wtout=${stout%.fits}_weight.fits

# Build command line:
suff=$(echo ${list%.lst} | cut -d\_ -f2 )
logfile=$WRK/images/pmerge_$suff.log               #logfile=$WRK/images/pmerge.log

# Command to produce the stack and its weight ...
args=" -c $confdir/swarp238.conf  -WEIGHT_SUFFIX _weight.fits -WEIGHT_TYPE MAP_WEIGHT  \
       -IMAGEOUT_NAME $stout  -WEIGHTOUT_NAME $wtout \
       -COMBINE_TYPE WEIGHTED  -RESAMPLE N  -DELETE_TMPFILES N  \
       -SUBTRACT_BACK N  -WRITE_XML Y  -XML_NAME pmerge_$suff.xml "
#ec "# Output stack: $stout"
#ec "# Command args:  $args"

comm="swarp @$list $args $verb"


ec "#------------------------------------------------------------------" | tee $logfile 
ec " >>>>  Merge $(cat $list | wc -l) substacks from $list  <<<<"        | tee -a $logfile 
ec "#------------------------------------------------------------------" | tee -a $logfile 

echo "% $comm " >> $logfile
if [ $dry == 'F' ]; then 
	$comm >> $logfile 2>&1
else
	echo $comm 
fi

#### DON'T BUILD THIS MASK HERE in DR6 - DONE ELSEWHERE  ####
###-----------------------------------------------------------------------------
### For pass 1, also build mask file; Command is mask_for_stack.py ...
###-----------------------------------------------------------------------------
##if [ $pass -eq 99 ]; then   
##    ec "#------------------------------------------------------------------" | tee -a $logfile 
##    ec " >>>>  Build mask etc. for $stout  <<<<"                             | tee -a $logfile 
##    ec "#------------------------------------------------------------------" | tee -a $logfile 
##    
##    # add the --extendedobj option to use back_size 512 / back_filtersize 5 in order to
##    # improve mask of bright star haloes - AM 24.jun.18
##    # Threshold of 1.0 seems ok for N and Y, not sure for others ....
##    
##    case $FILTER in
##        N | Y | NB | NB118 ) thr=1. ;;
##        J         ) thr=0.7 ;;
##        H         ) thr=0.5 ;;
##        K | Ks    ) thr=0.9 ;;
##    esac
##    
##	#  --script-path $confdir/c_script 
##    args=" --conf-path $confdir --extendedobj --threshold $thr "
##    comm="python $pydir/mask_for_stack.py -I $stout -W $wtout $args "
##    ec "% $comm "  | tee -a $logfile  ; ec ""
##    if [ $dry == 'F' ]; then 
##		$comm >> $logfile 2>&1 
##	fi
##fi

edate=$(date "+%s"); dt=$(($edate - $sdate))
ec "#------------------------------------------------------------------" | tee -a $logfile 
ec " >>>> pmerge finished - walltime: $dt sec  <<<<"                     | tee -a $logfile 
ec "#------------------------------------------------------------------" | tee -a $logfile 
exit $errcode

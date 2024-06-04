#!/bin/sh
#PBS -S /bin/sh
#PBS -N Masks_@FILTER@_@ID@
#PBS -o @IDENT@.out            
#PBS -j oe
#PBS -l nodes=1:ppn=5,walltime=24:00:00
#-----------------------------------------------------------------------------
# module: mkMasks.sh - wrapper to launch mkMasks.py
# requires: intelpython, uvis scripts mkMasks.py and meanLevels.py
# inputs: 
# - list of images: of the form {name}_??.lst
# procedure: 
#   used to build ad-hoc scripts mkMasks_??.sh. They create a new subdir
#   named mkMasks_??, copy into it the needed data and files, and run mkMasks.py.
#   On terminating, the final products are copied back to the main directory and
#   the directory is deleted
# outputs:
# - a mask file for each image of the input list,
# - logfiles mkMasks_?.log (and .err), and mkMasks_{FILTER}_??.out from torque
# Notes:
# - requires ~32 GB for a batch of 100 frames, of which 26 are the _mask files
#   produced ==> can run on small scratch disks
#-----------------------------------------------------------------------------
# apr.23: minor adaptations for DR6
# - input head files in .../images/heads_q1m
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
ecn() { echo -n "$(date "+[%d.%h.%y %T"]) $1 "; }    # echo with date
mycd() { \cd $1; ec " --> $PWD"; }               # cd with message
sdate=$(date "+%s")

uvis=/home/moneti/softs/uvis-pipe    # top UltraVista code dir
bindir=$uvis/bin                     # pipeline modules
pydir=$uvis/python                   # python scripts
confdir=$uvis/config                 # config dir
errcode=0

#-----------------------------------------------------------------------------
# Setup
#-----------------------------------------------------------------------------
module=mkMasks                    # w/o .sh extension

# check  if run via shell or via qsub: 
if [[ "$0" =~ "$module" ]]; then
    echo "### This is $module: running as shell script ###"
    list=$1
    WRK=$WRK
    FILTER=$FILTER
    if [[ "${@: -1}" =~ 'dry' ]]; then dry=T; else dry=F; fi
else
    echo "### This is $module: running via qsub ###"
    dry=@DRY@
    list=@LIST@
    FILTER=@FILTER@
    WRK=@WRK@
fi

## mostly for testing purposes - used in shell mode
#if [ $# -eq 2 ]; then 
#    if [ $2 != 'dry' ]; then    
#        thresh=" --threshold $2 "
#        osuff="_mask_$2.fits"
#    fi
#fi

#-----------------------------------------------------------------------------
# The REAL work ... done in temporary directory
#-----------------------------------------------------------------------------

datadir=$WRK/images              # reference dir
rhost=$(echo $WRK | cut -c 2-4)  # host of $WRK

# build work dir name: 
dirname=$(echo $list | cut -d\. -f1)
whost=$(hostname)   #; echo "DEBUG: ref/work hosts: $rhost  $whost"

if [[ $whost == 'n09' ]] || [[ $whost == 'n08' ]] || [[ $whost == 'n17' ]]; then
    workdir=/${whost}data/${dirname}_$FILTER     # node with small scratch
else                        
    workdir=/scratch/${dirname}_$FILTER          # other node
fi

if [ ! -d $datadir ];       then echo "ERROR: $WRK/images not found ... quitting"; exit 5; fi
if [ ! -s $datadir/$list ]; then echo "ERROR: $list not found in $WRK/images ... quitting"; exit 5; fi

nl=$(cat $datadir/$list | wc -l)
ec "=================================================================="
ec " >>>>  Begin mkMasks on $list with $nl entries   <<<<"
ec "------------------------------------------------------------------"

# command line
osuff="_mask.fits"
# root for reference files for DR6; these are links to the DR5 full stack,
# its weight and its obFlag
stout=@REFMASK@   #DR5_${FILTER}  

refs=" -S ${stout}  -M zeroes.fits  "
thresh=" --threshold 1.5 "   # in practice the default value
args=" --inweight-suffix _weight.fits  --outweight-suffix $osuff  $thresh  \
       --conf-path $confdir  -T 6 -v NORMAL "

comm="python $pydir/mkMasks.py -l $list  $refs $args"
logfile=$(echo $list | cut -d\. -f1)

#-----------------------------------------------------------------------------
# run these in a separate subdir to avoid problems with parallel runs
#-----------------------------------------------------------------------------

if [ -d $workdir ]; then rm -rf $workdir; fi
mkdir $workdir
if [ $? -ne 0 ]; then ec "ERROR: could not build $workdir - quitting"; exit 1; fi

mycd $workdir

ec "## Working on $(hostname); data on $rhost; work dir is $workdir"
ec "## Link the needed data and config files... "
cp $datadir/$list .
ln -s $datadir/${stout} .                    # the global obFlag

ec "## Command line is:"
ec "% $comm "
ec "#  where ${stout} is: "
ecn "#  "; ls -l ${stout} | tr -s \  | cut -d\  -f9-11
ec ""      ;#echo drt = $dry

if [ $dry == 'T' ]; then echo "   >> EXITING TEST MODE << "; exit 3; fi

rm -f zeroes.fits       # just in case, before dfining new one
lbpm=$(\ls -t /n08data/UltraVista/DR6/bpms/bpm*201902*.fits | head -1)
ln -sf $lbpm zeroes.fits
cp $confdir/missfits.conf .

for f in $(cat $list); do r=${f%.fits}
    ln -s $datadir/origs/$f .                  # should be CASU image
    ln -s $datadir/weights/${r}_weight.fits .
    ln -s $datadir/heads_q1m/${r}.head .
done

#-----------------------------------------------------------------------------
# Check links:
ec "## Input file links are like:"
imroot=$(head -1 $list | cut -d \. -f1)
ls -lh ${imroot}*.* | tr -s ' ' | cut -d ' ' -f9-12 

nims=$(ls -L v20*_0????.fits | wc -l)
nhds=$(ls -L v20*_0????.head | wc -l)
nwgs=$(ls -L v20*_0????_weight.fits | wc -l)

ec "## Build links to $nims images, $nwgs weights, $nhds heads"
if [[ $nims -ne $nwgs ]] || [[ $nims -ne $nhds ]]; then
    ec "PROBLEM: $nims, $nwgs, $nhds not all equal ... quitting"
    exit 5
fi
#-----------------------------------------------------------------------------

echo "% $comm >> $logfile.log "
if [ $dry != 'T' ]; then
    ec "## Begin work ... "
    $comm >> $logfile.log 2>> $logfile.err

    ec "# check products ..."
    nn=$(ls v20*${osuff} 2> /dev/null | wc -l)
    if [ $nn -lt $nl ]; then 
        echo "!!! PROBLEM: only $nn masks found for $nl images in input list"
    fi

	# remove blank lines from err file:
	grep -v "WARNING: FITS header" $logfile.err | strings  > x 
	if $(cat x | wc -l) -gt 0; then
		mv x $logfile.err
		ec "## Errors found; see $logfile.err)"
	else
		ec "# No errors found ... good job"
		rm $logfile.err
	fi

    ec "# and build the masks levels file ..."
    echo "# Build masks levels file: $logfile.dat" >> $logfile.log
    \ls -1 v20*$osuff > mlist
    python $pydir/validFraction.py -l mlist  > ${logfile}.dat
    rm mlist

    echo " >> move $nn masks back to images/ and clean up" >> $logfile.log
    mv v20*_*${osuff}  mkMasks_??.* $logfile*  $datadir/
    rm v20*_0????.fits v20*_00???_weight.fits v20*_00???.head   # the links
else
    echo "###  exit dry mode"
    exit 0
fi

cd $datadir/
rm -rf $workdir

#-----------------------------------------------------------------------------
# and finish up
#-----------------------------------------------------------------------------
edate=$(date "+%s"); dt=$(($edate - $sdate))
echo " >>>> $module.sh finished - walltime: $dt sec  <<<<"
echo "------------------------------------------------------------------"

exit $errcode

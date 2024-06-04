#!/bin/sh
#PBS -S /bin/sh
#PBS -N subSky_@FILTER@_@ID@
#PBS -o @IDENT@.out            
#PBS -j oe
#PBS -l nodes=1:ppn=5,walltime=18:00:00
#-----------------------------------------------------------------------------
# module: subSky wrapper for subSky.py
# requires: intelpython, astropy.io.fits, uvis scripts and libs
# Purpose: actual sky subtraction and destriping
#
# set ppn=5 to limit num jobs running in parallel to 8.  This also leaves 3
# nodes free for other stuff (interactive work).  
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
mycd() { \cd $1; echo " --> $PWD"; }               # cd with message
sdate=$(date "+%s")

uvis=/home/moneti/softs/uvis-pipe    # top UltraVista code dir
bindir=$uvis/bin                     # pipeline modules
pydir=$uvis/python                   # python scripts
confdir=$uvis/config                 # config dir
errcode=0

#-----------------------------------------------------------------------------
# Setup
#-----------------------------------------------------------------------------
module=subSky                     # w/o .sh extension

# check  if run via shell or via qsub: 
if [[ "$0" =~ "$module" ]]; then
    echo "$module: running as shell script "
    list=$1
    WRK=$WRK
    FILTER=$FILTER
	if [[ "${@: -1}" =~ 'dry' ]]; then dry=T; else dry=F; fi
else
    echo "$module: running via qsub (from pipeline)"
    dry=@DRY@
    list=@LIST@
    FILTER=@FILTER@
    WRK=@WRK@
fi

#-----------------------------------------------------------------------------
# The REAL work ... done in temporary workdir
#-----------------------------------------------------------------------------

datadir=$WRK/images            # reference dir
rhost=$(echo $WRK | cut -c 2-4)  # host of WRK

if [[ "$rhost" =~ "c0" ]]; then
	echo "### On login node $rhost ... not good ... quitting"
	exit 0
fi

# build work dir: 
dirname=$(echo $list | cut -d\. -f1)
whost=$(hostname)   #; echo "DEBUG: ref/work hosts: $rhost  $whost"

if [[ $whost == 'n09' ]] || [[ $whost == 'n08' ]] || [[ $whost == 'n17' ]]; then
	workdir=/${whost}data/${dirname}_$FILTER     # node with small scratch
else 
	workdir=/scratch/${dirname}_$FILTER          # other node
fi

if [ ! -d $datadir ];       then echo "ERROR: $WRK/images not found ... quitting"; exit 5; fi
if [ ! -s $datadir/$list ]; then echo "ERROR: $list not found in $WRK/images ... quitting"; exit 5; fi

case $FILTER in
	NB118 | Y | J  ) delta=40 ;;
	H | Ks | T | S ) delta=20 ;;
	Q | R | P      ) delta=40 ;;   # the test filters
esac   
		
nl=$(cat $datadir/$list | wc -l)

# run these in a separate subdir to avoid problems with parallel runs
if [ -d $workdir ]; then rm -rf $workdir/*; fi

mkdir $workdir 
mycd $workdir
if [[ $(hostname) =~ 'c' ]]; then
	echo "ERROR:  On login node - must go to compute node - quitting"
	exit 8
fi

ec "=================================================================="
ec " >>>>  Begin subSky with $list of $nl images  <<<<"
ec "------------------------------------------------------------------"
ec "## Working on $(hostname); data on $rhost; work dir is $workdir"

# command line
osuff="_clean"       # w/o .fits extension
args=" --inmask-suffix _mask.fits "
conf=" --config-path $confdir  --script-path $confdir  --outname-suffix $osuff "
comm="python $pydir/subSky.py -l $list $args $conf "


# copy or link needed input files
cp $datadir/$list .
for f in $(cat $list); do 
	r=${f%.fits}
	ln -s $datadir/withSky/${r}_withSky.fits ${r}.fits
	ln -s $datadir/weights/${r}_weight.fits .
	ln -s $datadir/mkSky/${r}_sky.fits .
	ln -s $datadir/Masks/${r}_mask.fits .
done

cp $confdir/bgsub.param $confdir/swarp238.conf .
cp $confdir/bgsub.conf $confdir/gauss_3.0_7x7.conv .

logfile=$(echo $list | cut -d\. -f1).log

#-----------------------------------------------------------------------------
# Check links:
ec "## Input file links are like:"
imroot=$(head -1 $list | cut -d \. -f1)
ls -lh ${imroot}*.* | tr -s ' ' | cut -d ' ' -f9-12 

nims=$(ls -L v20*_0????.fits | wc -l)           # with sky images
nwgs=$(ls -L v20*_0????_weight.fits | wc -l)    # updated weights
nsky=$(ls -L v20*_0????_sky.fits  | wc -l)      # skies to subtract
nmks=$(ls -L v20*_0????_mask.fits | wc -l)      # masks

ec "## Built links to $nims images, $nwgs weights, $nsky skies, $nmks masks"
if [[ $nims -ne $nwgs ]] || [[ $nims -ne $nsky ]] || [[ $nims -ne $nmks ]]; then
	ec "PROBLEM: $nims, $nwgs, $nsky, $nmks not all equal ... quitting"
	exit 5
fi
#-----------------------------------------------------------------------------

ec "## Command line is: "
ec "% $comm >> $logfile 2>&1"

if [ $dry == 'T' ]; then
    ec " ## DRY MODE - do nothing - clean up and quit now  ## "
	rm -rf $workdir
	exit 0
else
	ec "-----------------------------------------------------------------------------" > $logfile
	ec " - logfile of ${logfile%.log}.sh  " >> $logfile
	ec "-----------------------------------------------------------------------------" >> $logfile
	ec " - found $list with $(cat $list | wc -l) entries" >> $logfile
	ec " - Command line is: "      >> $logfile
	ec " % $comm >> $logfile 2>&1" >> $logfile
	ec "-----------------------------------------------------------------------------" >> $logfile
	ec "" >> $logfile

	# do the work
    $comm | grep -v ^$ > $logfile 2>&1    # removing blank lines

	# check products: num files and file size
	# in DR6: products moved already to filter dir, so check wrt logfile
	grep Begin\ work $logfile > done.lst
	nsub=$(cat done.lst | wc -l)
	if [ $nsub -ne $nims ]; then
		ec "!!! PROBLEM: found only $nsub _clean files of $nims expected"
		errcode=4
	else
		ec " >> Found $nsub _clean files - move them back to \$WRK/images/cleaned and clean up"
		rsync -av v20*clean.fits $WRK/images/cleaned > mv.log
		nm=$(grep clean.fits mv.log | wc -l)
		ec " >> ... $nm _clean.fits files moved; ready to remove $workdir"
		mv $logfile $WRK/images
	fi
fi

cd $datadir
if [ $errcode -eq 0 ]; then rm -rf $workdir; fi
	
#-----------------------------------------------------------------------------
# and finish up
#-----------------------------------------------------------------------------
edate=$(date "+%s"); dt=$(($edate - $sdate))
ec " >>>> $module.sh finished - walltime: $dt sec  <<<<"
ec "------------------------------------------------------------------"
exit $errcode

#-----------------------------------------------------------------------------

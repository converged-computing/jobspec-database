#!/bin/bash
#PBS -N JuliaMill
#PBS -l select=1:ncpus=3:mem=100gb:scratch_local=50gb
#PBS -l walltime=55:00:00 
#PBS -m ae

SCRIPTNAME=analyser.jl
MY=""
SCRIPT=$MY/master/malware-analyses/$SCRIPTNAME
DATADIR=$MY/master/reports/data/
LOCAL=$MY/log
LOGFILE="job_info.log"
THREADS=2
SIGNATURE=""


echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $LOCAL/$LOGFILE
export MODULEPATH=$MODULEPATH:/my_modules

echo "LOADING MODUL" >> $LOCAL/$LOGFILE
module add julia-153

test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

echo "COPYING DATA" >> $LOCAL/$LOGFILE
cp -r $DATADIR.  $SCRATCHDIR/data/ || { echo >&2 "Error while copying input file(s)!"; exit 2; }

echo "COPYING SCRIPT" >> $LOCAL/$LOGFILE
cp $SCRIPT $SCRATCHDIR

echo "COPYING JULIA LIBS" >> $LOCAL/$LOGFILE
cp -r $MY/.julia $SCRATCHDIR

echo "CD TO SCRATCHDIR" >> $LOCAL/$LOGFILE
cd $SCRATCHDIR

mkdir out

echo "RUNNING JULIA" >> $LOCAL/$LOGFILE
julia -t $THREADS $SCRIPTNAME $SCRATCHDIR $SIGNATURE || { echo >&2 "Calculation ended up erroneously (with a code $?) !!"; exit 3; }

echo "COPYING RESULTS" >> $LOCAL/$LOGFILE
mkdir $LOCAL/$PBS_JOBID
cp -r out/. $LOCAL/$PBS_JOBID || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 4; }

clean_scratch
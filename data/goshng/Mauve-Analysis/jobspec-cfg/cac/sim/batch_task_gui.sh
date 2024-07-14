#!/bin/bash

PMI_RANK=$1
JOBIDFILE=$2
LOCKFILE=$3
STATUSDIR=$4
MAXNODE=$5

WHICHLINE=1
JOBID=0
LASTNODE=N

cd $TMPDIR

function hms
{
  s=$1
  h=$((s/3600))
  s=$((s-(h*3600)));
  m=$((s/60));
  s=$((s-(m*60)));
  printf "%02d:%02d:%02d\n" $h $m $s
}

# Keep trying to read in jobidfile until the current node is the last one.
while [ "$JOBID" != "" ]; do
  lockfile=$LOCKFILE
  if ( set -o noclobber; echo "$$" > "$lockfile") 2> /dev/null; then
    trap 'rm -f "$lockfile"; exit $?' INT TERM
    read -r JOBID < ${JOBIDFILE}
    sed '1d' $JOBIDFILE > $JOBIDFILE.temp; 
    mv $JOBIDFILE.temp $JOBIDFILE
    rm -f "$lockfile"
    trap - INT TERM

    if [ "$JOBID" == "" ]; then
      echo "No more job"
    else
      echo "Job: $JOBID"
      START_TIME=`date +%s`
      echo $JOBID | bash
      # checkIfTheJobSuccessfullyFinished
    fi
  else
    JOBID=0
    sleep 5
  fi
done


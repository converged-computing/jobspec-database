#!/bin/bash

#PBS -l select=5:ncpus=10:mem=50gb
#PBS -l place=pack:shared
#PBS -l pvmem=235gb
#PBS -l walltime=12:00:00

HOST=`hostname`
LOG="$STDOUT_DIR2/$HOST_getPatric_.log"
ERRORLOG="$STDERR_DIR2/$HOST_error.log"

if [ ! -f "$LOG" ] ; then
	touch "$LOG"
fi

echo "Started `date`">>"$LOG"

echo "Host `hostname`">>"$LOG"

#
# load tools
#
module load perl
module load blast

#
# get file
#

XFILE=`head -n +${PBS_ARRAY_INDEX} $FILES_LIST | tail -n 1`
CFILE=${XFILE:2}
FILE="$SPLIT/$CFILE"

echo "working on File \"$FILE\"" >>"$LOG"

#
# Prodigal detection of CDS 
#

export OUT="$PRODIGAL_DIR/$CFILE.gbk"
export PROT="$PRODIGAL_DIR/$CFILE.faa"

echo "$PRODIGAL -i $FILE -o $OUT -a $PROT -p meta" >>$LOG

$PRODIGAL -i $FILE -o $OUT -a $PROT -p meta

#
# remove contigs containing only 0 or 1 cds
#

export OUT2="$SELECTED/selected_$CFILE.faa"

export RUN="$WORKER_DIR/remove_simple_contigs.pl"
echo "$RUN $PROT $OUT2" >>$LOG

perl $RUN $PROT $OUT2

#
# run Uproc
#
export OUT3="$UPROC_DIR/uproc_$CFILE.txt"

export RUN="uproc-prot -p -t 10 -o $OUT3 -P 3 $DB_DIR $MODEL $OUT2"
echo "$RUN" >> $LOG 

$RUN
  

#
# recover potential AMGs
#
export OUT4="$UPROC_DIR/AMGs_$CFILE.txt"

export RUN="$WORKER_DIR/get_hits_uproc.pl"
echo "$RUN $OUT3 $OUT4 $PFAM_VIR"

perl $RUN $OUT3 $OUT4 $PFAM_VIR


echo "Finished `date`">>"$LOG"


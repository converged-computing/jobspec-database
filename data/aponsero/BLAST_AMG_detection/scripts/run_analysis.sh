#!/bin/bash

#PBS -l select=1:ncpus=10:mem=50gb
#PBS -l walltime=12:00:00
#PBS -l cput=120:00:00

LOG="$STDOUT_DIR2/analysis.log"
ERRORLOG="$STDERR_DIR2/analysis.log"

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
echo "$RUN $PROT $OUT2"

perl $RUN $PROT $OUT2

#
# run blast on bacterial Database
#

export OUT3="$BLAST_DIR/bact_$CFILE.txt"

echo "blastp -query $OUT2 -db $BACTERIAL_DB -out $OUT3 -outfmt 6 -num_threads 10 -max_hsps 1" >>$LOG

blastp -query $OUT2 -db $BACTERIAL_DB -out $OUT3 -outfmt 6 -num_threads 10 -max_hsps 1 

#
# run blast on viral Database
#

export OUT4="$BLAST_DIR/vir_$CFILE.txt"
echo "blastp -query $OUT2 -db $VIRAL_DB -out $OUT4 -outfmt 6 -num_threads 10 -max_hsps 1" >>$LOG

blastp -query $OUT2 -db $VIRAL_DB -out $OUT4 -outfmt 6 -num_threads 10 -max_hsps 1

#
# recover potential AMGs
#

export BACT="$SAMPLE_DIR/blast/bact_$CFILE.txt"
export VIR="$SAMPLE_DIR/blast/vir_$CFILE.txt"

export RUN="$WORKER_DIR/get_hits.pl"
export OUTRESULT="$SAMPLE_DIR/blast/results_$CFILE.tab"
perl $RUN $BACT $VIR $OUTRESULT $FIG_LOG $TAX_LOG 2> "ERRORLOG"




echo "Finished `date`">>"$LOG"


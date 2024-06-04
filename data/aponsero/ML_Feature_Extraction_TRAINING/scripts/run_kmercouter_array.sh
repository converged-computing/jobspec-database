#!/bin/bash

#PBS -l select=12:ncpus=1:mem=10gb
#PBS -l place=free:shared
#PBS -l walltime=6:00:00


HOST=`hostname`
LOG="$STDOUT_DIR2/$HOST_getKmerFreq_.log"
ERRORLOG="$STDERR_DIR2/$HOST_error.log"

if [ ! -f "$LOG" ] ; then
	touch "$LOG"
fi

echo "Started `date`">>"$LOG"

echo "Host `hostname`">>"$LOG"

cd $RESULT_DIR

export OUT_DIR="$RESULT_DIR/kmers"

if [ ! -d "$OUT_DIR" ] ; then
        mkdir "$OUT_DIR"
fi

MYFILE=`head -n +${PBS_ARRAY_INDEX} $FILES_LIST | tail -n 1`
FILE=${MYFILE:2}

export DB="$RESULT_DIR/$FILE"
export OUT="$OUT_DIR/kmer_$FILE.tab"


module load python/3/3.5.2

export RUN="$WORKER_DIR/kmer_freq.py"
python3 $RUN -f $DB -k $KM_SIZE -t $READTYPE > $OUT


echo "Finished `date`">>"$LOG"

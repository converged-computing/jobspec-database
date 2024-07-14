#!/bin/bash

OFFSET=$1

SAMPLE_NAME=merged_dlana_mcc9_v28_wctagger_run1_bnb_nu_overlay
FILTER_TYPE=numu-sideband
INPUTLIST=/cluster/tufts/wongjiradlab/larbys/run_dlana_jobs/filter_inputlist/${SAMPLE_NAME}.list
UBDLANA_DIR=/cluster/tufts/wongjiradlab/twongj01/production/ubdlana/
OUTPUT_DIR=/cluster/tufts/wongjiradlab/larbys/run_dlana_jobs/filtered_outputdir/${SAMPLE_NAME}
OUTPUT_LOGDIR=/cluster/tufts/wongjiradlab/larbys/run_dlana_jobs/filtered_logdir/${SAMPLE_NAME}

mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_LOGDIR

# GET LINE
stride=1
jobid=$(( ${SLURM_ARRAY_TASK_ID} + ${OFFSET} ))
#jobid=${SLURM_ARRAY_TASK_ID}
let startline=$(expr "${stride}*${jobid}")

# GET INPUT FILENAME
let lineno=${startline}+1
inputfile=`sed -n ${lineno}p ${INPUTLIST}`
echo "inputlist: $inputfile"
inputbase=$(basename ${inputfile})

# LOCAL JOBDIR
local_jobdir=`printf /tmp/dlfilter_${FILTER_TYPE}_jobid%03d_${SAMPLE_NAME} $jobid`
echo "local jobdir: $local_jobdir"
rm -rf $local_jobdir
mkdir -p $local_jobdir

# get fileno
fileno=`echo $inputbase | sed 's/.*[^0-9]\([0-9]\+\)[^0-9]*$/\1/'`

# local outfile
local_outfile=`printf dlfilter_${FILTER_TYPE}_${SAMPLE_NAME}_jobid%s.root $fileno`
echo "outfile : "$local_outfile

# local log file
local_logfile=`printf dlfilter_${FILTER_TYPE}_${SAMPLE_NAME}_jobid%s.log $fileno`
echo "output logfile: "$local_logfile

cd $UBDLANA_DIR
source setup_tufts_container.sh
source setup_env.sh

echo "PYTHONPATH:"
echo $PYTHONPATH

cd $local_jobdir
echo "python $UBDLANA_DIR/bin/run_ubdlfilter_lar.py -d $inputfile -c $UBDLANA_DIR/CalibrationMaps_MCC9.root -t BNB -f ${FILTER_TYPE} -o $local_outfile >& $local_logfile"
python $UBDLANA_DIR/bin/run_ubdlfilter_lar.py -d $inputfile -c $UBDLANA_DIR/CalibrationMaps_MCC9.root -t BNB -f ${FILTER_TYPE} -o $local_outfile >& $local_logfile

# move to output dir
cp $local_outfile $OUTPUT_DIR/
cp $local_logfile $OUTPUT_LOGDIR/

# clean-up
cd /tmp
rm -r $local_jobdir

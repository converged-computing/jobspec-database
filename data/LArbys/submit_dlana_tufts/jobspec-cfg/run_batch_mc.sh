#!/bin/bash

OFFSET=$1
STRIDE=$2
SAMPLE_NAME=$3

# we assume we are already in the container

INPUTLIST=/cluster/tufts/wongjiradlab/larbys/run_dlana_jobs/inputlists/${SAMPLE_NAME}.list
UBDLANA_DIR=/cluster/tufts/wongjiradlab/twongj01/production/ubdlana/
OUTPUT_DIR=/cluster/tufts/wongjiradlab/larbys/run_dlana_jobs/outputdir/${SAMPLE_NAME}
OUTPUT_LOGDIR=/cluster/tufts/wongjiradlab/larbys/run_dlana_jobs/logdir/${SAMPLE_NAME}

mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_LOGDIR

# WE WANT TO RUN MULTIPLE FILES PER JOB IN ORDER TO BE GRID EFFICIENT
start_jobid=$(( ${OFFSET} + ${SLURM_ARRAY_TASK_ID}*${STRIDE}  ))

# LOCAL JOBDIR
local_jobdir=`printf /tmp/dlana_jobid%04d_${SAMPLE_NAME} ${SLURM_ARRAY_TASK_ID}`
echo "local jobdir: $local_jobdir"
rm -rf $local_jobdir
mkdir -p $local_jobdir

# local log file
local_logfile=`printf merged_dlana_${SAMPLE_NAME}_jobid%04d.log ${SLURM_ARRAY_TASK_ID}`
echo "output logfile: "$local_logfile

echo "SETUP CONTAINER/ENVIRONMENT"
cd $UBDLANA_DIR
source setup_tufts_container.sh
source setup_env.sh

echo "GO TO JOBDIR"
cd $local_jobdir

echo "STARTING TASK ARRAY ${SLURM_ARRAY_TASK_ID} for ${SAMPLE_NAME}" > ${local_logfile}

# run a loop
for ((i=0;i<${STRIDE};i++)); do

    jobid=$(( ${start_jobid} + ${i} ))
    echo "JOBID ${jobid}"
  
    # GET INPUT FILENAME
    let lineno=${jobid}+1
    inputfile=`sed -n ${lineno}p ${INPUTLIST}`
    echo "inputfile path: $inputfile"

    # local outfile
    local_outfile=`printf merged_dlana_${SAMPLE_NAME}_jobid%04d.root $jobid`
    echo "outfile : "$local_outfile

    echo "python $UBDLANA_DIR/bin/run_ubdlana_lar.py -d $inputfile -c $UBDLANA_DIR/CalibrationMaps_MCC9.root -t Overlay --ismc -o $local_outfile"
    python $UBDLANA_DIR/bin/run_ubdlana_lar.py -d $inputfile -c $UBDLANA_DIR/CalibrationMaps_MCC9.root -t Overlay --ismc -oh ophitBeamCalib -o $local_outfile >> $local_logfile 2>&1

    # subfolder dir
    let nsubdir=${jobid}/100
    subdir=`printf %03d ${nsubdir}`

    # copy to subdir in order to keep number of files per folder less than 100. better for file system.
    echo "COPY output to "${OUTPUT_DIR}/${subdir}/
    mkdir -p $OUTPUT_DIR/${subdir}/
    cp ${local_outfile} $OUTPUT_DIR/${subdir}/
done

# copy log to logdir
cp $local_logfile $OUTPUT_LOGDIR/

# clean-up
cd /tmp
rm -r $local_jobdir

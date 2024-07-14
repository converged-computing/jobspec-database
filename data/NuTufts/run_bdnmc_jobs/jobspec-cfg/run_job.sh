#!/bin/bash

PARAM_FILE=$1
OUTPUT_DIR=$2
SCRIPT_DIR=$3

#BDNMC_DIR="/home/twongjirad/working/coherent/bdnmc"
BDNMC_DIR="/usr/local/bdnmc"

JOBID=${SLURM_ARRAY_TASK_ID}
#JOBID=65

# Make job dir
JOBDIR=`printf /tmp/bdnmc_job%03d $JOBID`
rm -rf $JOBDIR
mkdir -p $JOBDIR

# Define output dir
OUTDIR=`printf ${OUTPUT_DIR}/output_job%03d ${JOBID}`
mkdir -p $OUTDIR/

echo "JOBDIR: "$JOBDIR
echo "OUTDIR: "$OUTDIR

# Copy scripts we'll need
cp $SCRIPT_DIR/generate_config_file.py $JOBDIR/

# Go to job dir
cd $JOBDIR/


# make parameter file from template
config=`printf bdnmc_parameters_cenns750_jobid%03d.dat ${JOBID}`
python generate_config_file.py $PARAM_FILE $JOBID $config  $SCRIPT_DIR/parameter_file_cenns750_template.dat $JOBDIR
echo "Generated "$config

$BDNMC_DIR/build/./main $config >& log.dat
cp *.dat $OUTDIR/

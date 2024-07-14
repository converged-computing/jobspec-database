#!/bin/bash

PARAM_FILE=$1
OUTPUT_DIR=$2
SCRIPT_DIR=$3

source /usr/local/root/root-6.16.00/bin/thisroot.sh
export GLB_DIR=/usr/local/globes-3.2.17
export SNOWGLOBES=/usr/local/snowglobes
#export SNOWGLOBES=/cluster/tufts/wongjiradlab/twongj01/coherent/snowglobes

JOBID=${SLURM_ARRAY_TASK_ID}
#JOBID=65

# Make job dir
JOBDIR=`printf /tmp/snowglobes_job%04d $JOBID`
rm -rf $JOBDIR
mkdir -p $JOBDIR/out

# Define output dir
JOB_OUTDIR=`printf ${OUTPUT_DIR}/output_job%04d ${JOBID}`
mkdir -p $JOB_OUTDIR

echo "JOBDIR: "$JOBDIR
echo "OUTDIR: "$JOB_OUTDIR

# go into job dir
cd $JOBDIR

# Copy snowglobes data files we'll need
source $SNOWGLOBES/coherent/sterile/copy_data_dir.sh $SNOWGLOBES $JOBDIR/

# Copy scripts we need
cp $SNOWGLOBES/coherent/sterile/supernova.pl .
cp $SNOWGLOBES/coherent/sterile/*.py .
cp $SNOWGLOBES/coherent/sterile/gen_osc_rate.py .

python gen_osc_rate.py $PARAM_FILE $JOBID $SNOWGLOBES/coherent/sterile/fluxes/stpi.dat $JOBDIR $JOBDIR/out/ >& $JOBDIR/log.dat

echo "Transfer for OUTDIR: "$JOB_OUTDIR/

cp $JOBDIR/log.dat $JOB_OUTDIR/
cp $JOBDIR/out/* $JOB_OUTDIR/

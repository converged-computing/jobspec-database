#!/bin/bash

let processid=$1
let gpuid=$2
echo "Spawning job for processid=${processid}"

WORKDIR=/cluster/kappa/90-days-archive/wongjiradlab/grid_jobs/ssnet-tuftscluster-scripts
INPUTLISTS=${WORKDIR}/inputlists
JOBIDLIST=${WORKDIR}/rerunlist.txt
DATAFOLDER=/cluster/kappa/90-days-archive/wongjiradlab/larbys/data

OUTDIR=${DATAFOLDER}/comparison_samples/extbnb_wprecuts_reprocess/out_week10132017/ssnet_p09

mkdir -p $OUTDIR

rm -f log_ssnettufts_job.txt
echo "launching job=$i" && export SLURM_PROCID=$processid && cd ${WORKDIR} && ./run_gpu_job.sh ${WORKDIR} ${INPUTLISTS} ${OUTDIR} ${JOBIDLIST} ${gpuid} >> log_ssnettufts_job.txt 2>&1 &

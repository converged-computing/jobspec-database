#!/bin/bash

# ----------------------------------------------------------------------------------------
# --- Run estimator in parallel on all pseudo-observed data
# ----------------------------------------------------------------------------------------

module load parallel

PARAM_d=$1
PARAM_s=$2
PARAM_i=$3

# Sample size of PODs to analyze
SAMPLE=$4

DSI_STRING=DNA${PARAM_d}_STR${PARAM_s}_IND${PARAM_i}

SUMSTATS=results/simulated_data
SUMSTATS=${SUMSTATS}/ABCsampler_output_${DSI_STRING}.sumstats.combined.txt

mkdir -p results/estimator_output
OUT_FILE=results/estimator_output/ABCestimator.$DSI_STRING.results.txt

TOTAL_PODS=`wc -l $SUMSTATS | cut -d' ' -f1`
TOTAL_PODS=$((TOTAL_PODS - 1))

seq 1 $TOTAL_PODS | shuf | head -n $SAMPLE | \
    parallel --jobs 20 \
        sh scripts/run_estimator.sh $PARAM_d $PARAM_s $PARAM_i {} > \
    $OUT_FILE

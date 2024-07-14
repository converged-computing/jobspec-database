#!/bin/bash
# check if script is started via SLURM or bash
# if with SLURM: there variable '$SLURM_JOB_ID' will exist
if [ -n "$SLURM_JOB_ID" ];  then
    export THREADS="$SLURM_CPUS_PER_TASK"
    export MEMORY="$SLURM_MEM_PER_NODE"
else
    export THREADS=4
    export MEMORY=4096
fi
# The location of this repo
export REPO_DIR=/home/staff/haukurpj/SMT

# Don't use language endings.
export LANGS="en is"

# Mosesdecoder
export MOSESDECODER=$REPO_DIR/../mosesdecoder

# Data locations
# In these folders a symbolic link should refer to the actual data
export FORMATTED_DIR="$REPO_DIR"/data/formatted
export RAW_DIR="$REPO_DIR"/data/raw
export OUT_DIR="$REPO_DIR"/data/out

# The location of the actual data and work directory, for volume mapping
export WORK_DIR=/work/haukurpj

# Truecasing - Where to write the model
export TRUECASE_MODEL=preprocessing/preprocessing/resources/truecase-model

# Model locations - Use a symbolic link
export MODEL_DIR="$REPO_DIR"/model
export EN_IS_DIR="$MODEL_DIR"/en-is
export IS_EN_DIR="$MODEL_DIR"/is-en
export MODEL_RESULTS_DIR="$MODEL_DIR"/results

# LM location and order
export LM_MODEL="$OUT_DIR"/blm
export LM_ORDER=5

# Location of data.
export TRAINING_DATA="$OUT_DIR"/train/data
export DEV_DATA="$OUT_DIR"/dev/data
export TEST_DIR="$OUT_DIR"/test

# Name of test sets.
export TEST_SETS="ees ema opensubtitles"
export TEST_SET_CORRECT_COMBINED="$TEST_DIR"/combined."$LANG_TO"
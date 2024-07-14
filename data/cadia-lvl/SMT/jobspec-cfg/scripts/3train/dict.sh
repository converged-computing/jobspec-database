#!/bin/bash
# e=fail on pipeline, u=fail on unset var, x=trace commands
set -ex

# check if script is started via SLURM or bash
# if with SLURM: there variable '$SLURM_JOB_ID' will exist
if [ -n "$SLURM_JOB_ID" ];  then
    export THREADS="$SLURM_CPUS_PER_TASK"
    export MEMORY="$SLURM_MEM_PER_NODE"
else
    export THREADS=4
    export MEMORY=4096
fi
# Steps: 1=Prepare 2= 3=Train 4=Tune 5=Binarise 
FIRST_STEP=4
LAST_STEP=5

# Model variables
LANG_FROM="$1"
LANG_TO="$2"
DIRECTION_DIR="$3"

CLEAN_MIN_LENGTH=1
CLEAN_MAX_LENGTH=70
ALIGNMENT="grow-diag"
REORDERING="msd-bidirectional-fe"

# Test if data is there
function check_data() {
  if [[ ! -f "$1" ]]; then
    echo "$1 does not exist. Exiting..."
    exit 1
  else
    wc -l "$1" 
    ls -hl "$1"
  fi
}

function check_dir_not_empty() {
  if [[ ! $(ls -A "$1") ]]; then
    echo "$1 is Empty"
    exit 1
  fi
}
check_data "$TRAINING_DATA"."$LANG_TO"
check_data "$TRAINING_DATA"."$LANG_FROM"
check_data "$DEV_DATA"."$LANG_FROM"
check_data "$DEV_DATA"."$LANG_TO"
check_dir_not_empty "$TEST_DIR"/
check_data "$LM_MODEL"."$LANG_TO"

# Variables for subsequent steps.
# 1. Prepare
MODEL_DATA_DIR="$DIRECTION_DIR"/data
CLEAN_DATA="$MODEL_DATA_DIR"/train

# 3. Train Moses
BASE_DIR="$DIRECTION_DIR"/base
BASE_MOSES_INI="$BASE_DIR"/model/moses.ini
BASE_PHRASE_TABLE="$BASE_DIR"/model/phrase-table.gz
BASE_REORDERING_TABLE="$BASE_DIR"/model/reordering-table.wbe-msd-bidirectional-fe.gz

# 4. Tune Moses
TUNE_DIR="$DIRECTION_DIR"/tuned
TUNED_MOSES_INI="$TUNE_DIR"/moses.ini

# 5. Binarise Moses
BINARISED_DIR="$DIRECTION_DIR"/binarised

# Step=1. This script prepares all directories and cleans data
if ((FIRST_STEP <= 1 && LAST_STEP >= 1)); then
  mkdir -p "$MODEL_DATA_DIR"
  # Data prep
  "$MOSESDECODER"/scripts/training/clean-corpus-n.perl "$TRAINING_DATA" "$LANG_FROM" "$LANG_TO" "$CLEAN_DATA" "$CLEAN_MIN_LENGTH" "$CLEAN_MAX_LENGTH"
fi

if ((FIRST_STEP <= 2 && LAST_STEP >= 2)); then
  echo "Not training LM here"
fi

# Step=3. Train Moses model
if ((FIRST_STEP <= 3 && LAST_STEP >= 3)); then
  mkdir -p "$BASE_DIR"
  "$MOSESDECODER"/scripts/training/train-model.perl \
    -root-dir "$BASE_DIR" \
    -corpus "$CLEAN_DATA" \
    -f "$LANG_FROM" \
    -e "$LANG_TO" \
    -alignment "$ALIGNMENT" \
    -reordering "$REORDERING" \
    -lm 0:"$LM_ORDER":"$LM_MODEL"."$LANG_TO":8 \
    -mgiza \
    -mgiza-cpus "$THREADS" \
    -parallel \
    -sort-buffer-size "$MEMORY" \
    -sort-batch-size 1021 \
    -sort-compress gzip \
    -sort-parallel "$THREADS" \
    -cores "$THREADS" \
    -external-bin-dir "$MOSESDECODER_TOOLS"
fi

# Step=4 Tuning
if ((FIRST_STEP <= 4 && LAST_STEP >= 4)); then
  mkdir -p "$TUNE_DIR"
  # When tuning over factors, it is best to skip the filtering.
  "$MOSESDECODER"/scripts/training/mert-moses.pl \
    "$DEV_DATA"."$LANG_FROM" \
    "$DEV_DATA"."$LANG_TO" \
    "$MOSESDECODER"/bin/moses "$BASE_MOSES_INI" \
    --mertdir "$MOSESDECODER"/bin \
    --working-dir "$TUNE_DIR" \
    --decoder-flags="-threads $THREADS"
fi

# Step=5. Binarise
function binarise_table() {
  PHRASE_TABLE_IN=$1
  BINARISED_PHRASE_TABLE_OUT=$2
  REORDERING_TABLE_IN=$3
  BINARISED_REORDERING_TABLE_OUT=$4

  "$MOSESDECODER"/bin/processPhraseTableMin \
    -in "$PHRASE_TABLE_IN" \
    -nscores 4 \
    -out "$BINARISED_PHRASE_TABLE_OUT" \
    -threads "$THREADS"

  "$MOSESDECODER"/bin/processLexicalTableMin \
    -in "$REORDERING_TABLE_IN" \
    -out "$BINARISED_REORDERING_TABLE_OUT" \
    -threads "$THREADS"
}

function fix_paths() {
  PATH_IN=$1
  PATH_OUT=$2
  FILE=$3
  sed -i "s|$PATH_IN|$PATH_OUT|" "$FILE"
  # Adjust the path in the moses.ini file to point to the new files.
}

if ((FIRST_STEP <= 5 && LAST_STEP >= 5)); then
  mkdir -p "$BINARISED_DIR"
  # TODO: Use many LMs
  BINARISED_LM="$BINARISED_DIR"/lm.blm
  BINARISED_MOSES_INI="$BINARISED_DIR"/moses.ini
  BINARISED_PHRASE_TABLE="$BINARISED_DIR"/phrase-table
  BINARISED_REORDERING_TABLE="$BINARISED_DIR"/reordering-table
  binarise_table "$BASE_PHRASE_TABLE" "$BINARISED_PHRASE_TABLE" "$BASE_REORDERING_TABLE" "$BINARISED_REORDERING_TABLE"
  cp "$LM_MODEL"."$LANG_TO" "$BINARISED_LM"
  cp "$TUNED_MOSES_INI" "$BINARISED_MOSES_INI"
  fix_paths "$LM_MODEL"."$LANG_TO" "$BINARISED_LM" "$BINARISED_MOSES_INI"
  fix_paths "$BASE_PHRASE_TABLE" "$BINARISED_PHRASE_TABLE" "$BINARISED_MOSES_INI"
  fix_paths "$BASE_REORDERING_TABLE" "$BINARISED_REORDERING_TABLE" "$BINARISED_MOSES_INI"
  sed -i "s|PhraseDictionaryMemory|PhraseDictionaryCompact|" "$BINARISED_MOSES_INI"
fi

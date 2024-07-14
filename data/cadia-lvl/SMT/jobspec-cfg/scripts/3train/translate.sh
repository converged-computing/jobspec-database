#!/bin/bash
# e=fail on pipeline, u=fail on unset var, x=trace commands
set -exx

# check if script is started via SLURM or bash
# if with SLURM: there variable '$SLURM_JOB_ID' will exist
if [ -n "$SLURM_JOB_ID" ];  then
    export THREADS="$SLURM_CPUS_PER_TASK"
    export MEMORY="$SLURM_MEM_PER_NODE"
else
    export THREADS=4
    export MEMORY=4096
fi

BINARISED_MOSES_INI="$1"
IN="$2"
OUT="$3"

# Translate
/opt/moses/bin/moses -f "$BINARISED_MOSES_INI" \
  -threads "$THREADS" \
  < "$IN" \
  > "$OUT"

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

TRANSLATED="$1"
REFERENCE="$2"
OUT="$3"

# Evaluate
/opt/moses/scripts/generic/multi-bleu-detok.perl -lc \
  "$TRANSLATED" \
    < "$REFERENCE" \
    > "$OUT"

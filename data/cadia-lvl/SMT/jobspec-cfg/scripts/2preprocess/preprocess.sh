#!/bin/bash
#SBATCH --job-name=moses-preprocess
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --partition=longrunning
#SBATCH --chdir=/home/staff/haukurpj/SMT
#SBATCH --time=2:01:00
#SBATCH --output=logs/%x-%j.out
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

LANG="$1"

function train_truecase() {
    LANG=$1
    cat "$OUT_DIR"/train+dict."$LANG" "$FORMATTED_DIR"/mono/data-short."$LANG" | \
    preprocessing/main.py tokenize - "$OUT_DIR"/truecase-data."$LANG" "$LANG" --threads "$THREADS" --batch_size 5000000 --chunksize 10000
    # We just use the truecaser from Moses, sacremoses is not good for this.
    "$MOSESDECODER"/scripts/recaser/train-truecaser.perl --model "$TRUECASE_MODEL"."$LANG" --corpus "$OUT_DIR"/truecase-data."$LANG"
}

# Add the dictionary data to the training data
cat "$FORMATTED_DIR"/parice/train."$LANG" "$FORMATTED_DIR"/dictionary/*."$LANG" > "$OUT_DIR"/train+dict."$LANG"
train_truecase "$LANG"
preprocessing/main.py preprocess "$FORMATTED_DIR"/mono/data-short."$LANG" "$OUT_DIR"/mono."$LANG" "$LANG" --truecase_model "$TRUECASE_MODEL"."$LANG" --threads "$THREADS" --batch_size 5000000 --chunksize 10000
preprocessing/main.py preprocess "$OUT_DIR"/train+dict."$LANG" "$TRAINING_DATA"."$LANG" "$LANG" --truecase_model "$TRUECASE_MODEL"."$LANG" --threads "$THREADS" --batch_size 5000000 --chunksize 10000
preprocessing/main.py preprocess "$FORMATTED_DIR"/parice/dev."$LANG" "$DEV_DATA"."$LANG" "$LANG" --truecase_model "$TRUECASE_MODEL"."$LANG" --threads "$THREADS" --batch_size 5000000 --chunksize 10000
# Data for LM training
cat "$TRAINING_DATA"."$LANG" "$OUT_DIR"/mono."$LANG" > "$OUT_DIR"/lm-data."$LANG"
bash scripts/run_in_singularity.sh scripts/2preprocess/lm.sh is "$OUT_DIR"/lm-data."$LANG" "$LM_MODEL"."$LANG" "$LM_ORDER" 
# The test data
rm "$TEST_DIR"/combined."$LANG" || true
rm "$TEST_DIR"/combined-processed."$LANG" || true
for TEST in $TEST_SETS; do
    cp "$FORMATTED_DIR"/parice/test-"$TEST"."$LANG" "$TEST_DIR"/"$TEST"."$LANG"
    preprocessing/main.py preprocess "$TEST_DIR"/"$TEST"."$LANG" "$TEST_DIR"/"$TEST"-processed."$LANG" "$LANG" --truecase_model "$TRUECASE_MODEL"."$LANG" --threads "$THREADS" --batch_size 5000000 --chunksize 10000
    cat "$TEST_DIR"/"$TEST"."$LANG" >> "$TEST_DIR"/combined."$LANG"
    cat "$TEST_DIR"/"$TEST"-processed."$LANG" >> "$TEST_DIR"/combined-processed."$LANG"
done
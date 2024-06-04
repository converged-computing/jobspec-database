#!/bin/bash --login
#PBS -l walltime=06:00:00
#PBS -A argonne_tpc
#PBS -q prod
#PBS -l select=48
#PBS -l filesystems=eagle:home
#

function sourceFile() {
    fp="$1"
    if [[ -f "${fp}" ]]; then
        echo "Found ${fp}, \`source\`-ing"
        # shellcheck source="${fp}"
        source "${fp}"
    else
        echo "ERROR: UNABLE TO SOURCE ${fp}"
    fi
}

# +++++++++++++++ SCRIPT START ++++++++++++++++++++++
# ---- source ./helpers_alcf.sh ---------------------
cd "${PBS_O_WORKDIR}" || exit
HERE=$(python3 -c 'import os; print(os.getcwd())')
sourceFile "${HERE}/ALCF_utils/helpers_alcf.sh" || exit
# cd ~/anl_24_release_q4/llm.devkit/Megatron-DeepSpeed || exit
# eval "$(/home/foremans/miniconda3/bin/conda shell.zsh hook)" && conda activate anl_release_q4v2
ezpz || exit
setEnv || exit
saveDSenv || exit
makeHostfiles || exit
setupData "${DATA_FILE_LIST:-${HERE}/data_file_list_reweighted.txt}" || exit
# dfl_fallback="${HERE}/data_file_list_shuf_debug.txt"

# # ---- DATA SETUP ------------------------------------
# dfl_debug="./data_file_list_shuf_debug.txt"
# DATA_FILE_LIST="${DATA_FILE_LIST:-${dfl_debug}}" && export DATA_FILE_LIST="${DATA_FILE_LIST}"
# NUM_DOCS=$(wc -l < "${DATA_FILE_LIST}") && export NUM_DOCS="${NUM_DOCS}"
# WEIGHT_SUM="$(sumWeights "${DATA_FILE_LIST}")" && export WEIGHT_SUM="${WEIGHT_SUM}"
# DFL_STEM=$(echo "$DATA_FILE_LIST" | tr "\/" "\t" | awk '{print $NF}' | sed "s/\.txt//g") && export DFL_STEM="${DFL_STEM}"
# dcp="${HERE}/.cache/${DFL_STEM}-index-cache"
# DATA_CACHE_PATH="${DATA_CACHE_PATH:-${dcp}}" && export DATA_CACHE_PATH="${DATA_CACHE_PATH}"
# mkdir -p "${DATA_CACHE_PATH}"
# if [[ -n "${DOLMA_CHUNK_IDX}" ]]; then
#     echo "Using DOLMA CHUNK ${DOLMA_CHUNK_IDX} from ${DATA_FILE_LIST} with ${NUM_DOCS} documents..."
# else
#     echo "Using NUM_DOCS=${NUM_DOCS} documents from DATA_FILE_LIST=${DATA_FILE_LIST}"
# fi


# ---- Parallelism Settings --------------------------
PP=${PP:-1}
TP=${TP:-1}
export PP="${PP}"
export TP="${TP}"
export HOSTFILE="${HOSTFILE:-${PBS_NODEFILE}}"
export WORLD_SIZE=${WORLD_SIZE:-$(wc -l < "${HOSTFILE}")}
# export WORLD_SIZE=${WORLD_SIZE:-$(wc -l < "${PBS_NODEFILE}")}
# ----------------------------------------------------

# ---- Llama2 7B Config -----------------------
export HEADS=${HEADS:-32}
export NLAYERS=${NLAYERS:-32}
export HIDDEN=${HIDDEN:-4096}
export NUM_KV_HEAD=${NUM_KV_HEAD:-8}
export MODEL_TYPE="llama-seq${SEQ}-pp${PP}-tp${TP}-${NLAYERS}layers-${HEADS}heads-${HIDDEN}hidden"
# ---------------------------------------------

# ---- Run Settings ---------------------------
export LR=${LR:-0.0003}
export SEQ=${SEQ:-4096}
export DTYPE=${DTYPE:-bf16}
export ZERO_STAGE=${ZERO_STAGE:-2}
export MICRO_BATCH=${MICRO_BATCH:-4}
export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}
export TRAIN_ITER=${TRAIN_ITER:-317892}
export SAVE_INTERVAL=${SAVE_INTERVAL:-200}
export USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING:-1}
export GLOBAL_BATCH=$(( $WORLD_SIZE * $MICRO_BATCH * $GRAD_ACC_STEPS / $TP / $PP ))
export USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING:-0}
export TOKENIZER_MODEL="/lus/gecko/projects/Aurora_deployment/AuroraGPT/datasets/dolma/utils/tokenizer.model"
# export EXTRA_ARGS=""
export LLAMA_ARGS="--no-query-key-layer-scaling --use-rotary-position-embeddings --untie-embeddings-and-output-weights --swiglu --normalization rmsnorm --disable-bias-linear"
# ---------------------------------------------

# ---- Build DeepSpeed Config ---------------------------------
export DS_CONFIG="ds_stage${ZERO_STAGE}_mb${MICRO_BATCH}_gb${GLOBAL_BATCH}_pp${PP}_${DTYPE}.json"
bash "${HERE}/generate_config.sh" "${DS_CONFIG}" || exit
# -------------------------------------------------------------


# ---- Specify output location --------------------------------
export OUTPUT_PREFIX="ds_stage${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_mb${MICRO_BATCH}_seq${SEQ}_gb${GLOBAL_BATCH}_pp${PP}_tp${TP}_${DTYPE}"
# OUTPUT_DIR=logs/ds_stage${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_mb${MICRO_BATCH}_seq${SEQ}_gb${GLOBAL_BATCH}_pp${PP}_tp${TP}_${DTYPE}_`date +%m%d%H%M%S`_${HOSTNAME}
OUTPUT_DIR="logs/${OUTPUT_PREFIX}/$(date +%m%d%H%M%S)_${HOSTNAME}"
export OUTPUT_DIR="${OUTPUT_DIR}"
export OUTPUT_LOG="${OUTPUT_DIR}/output.log"
export CKPT_DIR="checkpoints/${OUTPUT_PREFIX}"
echo "${OUTPUT_LOG}" >> "logs/latest"
mkdir -p "${OUTPUT_DIR}"
echo "!!!Please see logs at ${OUTPUT_DIR}"


gpt_args=()
ds_args=" "
ds_args=" --deepspeed ${ds_args}"
if [ "$PP" == 1 ]; then
   ds_args=" --no-pipeline-parallel ${ds_args}" 
fi
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"

# BUG: [???] ----------------------------------------------------------------
# I dont know where this came from...
# > we are now using activation checkpoint provided by megatron, see below.
# ---------------------------------------------------------------------------
#
# NOTE: [???] ---------------------------------------------------------------
# In `train_llama_alcf_polaris.sh` we also pass
# `"--checkpoint-num-layers 1"`
# ----------------------------------------------------------------------------
if [[ "$USE_ACTIVATION_CHECKPOINTING" == 1 ]]; then
    echo "!! Caught USE_ACTIVATION_CHECKPOINTING=${USE_ACTIVATION_CHECKPOINTING} !!"
    ds_args=" --deepspeed-activation-checkpointing ${ds_args}"
    gpt_args+=(
        "--checkpoint-activations"
    )
    # "--checkpoint-num-layers 1"
    # --checkpoint-activations \
    # --deepspeed-activation-checkpointing
fi

# take custom args
custom_args=" $@"

# Ensure `./hostfile_deepspeed` and `./hostfile_mpich` exist in $(pwd)
hfds="${HERE}/hostfile_deepspeed"
hfmpi="${HERE}/hostfile_mpich"
[ -f "$hfds" ] || exit
[ -f "$hfmpi" ] || exit

# launcher setting
LAUNCHER=${LAUNCHER:-MPICH}
if [[ $LAUNCHER == "deepspeed" ]]; then
    launcher=""
else
    launcher="--force_multi --hostfile ${hfds} --launcher=${LAUNCHER} --launcher_args='-hostfile ${hfmpi}'"
fi


if [[ $(hostname) == x4* ]]; then
    CCL=${CCL:-ccl}
    BE="${CCL}"
elif [[ $(hostname) == x3* ]]; then
    NCCL=${NCCL:-nccl}
    BE="${NCCL}"
fi
# NCCL=${NCCL:-nccl}
EXEC=pretrain_gpt_alcf.py

# MODEL=LLAMA_7B
# OUTPUT_PREFIX=${MODEL}_z${ZERO_STAGE}_seqlen_tp${TP}_pp${PP}_sp${SP}_nl${NUM_LAYERS}_hs${HIDDEN_SIZE}_gb${BS}_mb${MBS}
echo "++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "- WORLD_SIZE:${WORLD_SIZE}"
echo "- BACKEND: ${BE}"
echo "- MODEL_TYPE: ${MODEL_TYPE}"
echo "- DOCUMENT WEIGHT_SUM: ${WEIGHT_SUM}"
echo "- Using DATA_FILE_LIST: ${DATA_FILE_LIST}"
echo "- Using NUM_DOCS=${NUM_DOCS} documents from DATA_FILE_LIST=${DATA_FILE_LIST}"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++"

run_cmd="
    deepspeed $launcher ${EXEC} \
    --use-flash-attn \
    --num-key-value-heads ${NUM_KV_HEAD} \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NLAYERS \
    --hidden-size $HIDDEN \
    --num-attention-heads $HEADS \
    --seq-length $SEQ \
    --max-position-embeddings $SEQ \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --train-iters $TRAIN_ITER \
    --lr ${LR} \
    --lr-decay-style cosine \
    --log-interval 1 \
    --save-interval ${SAVE_INTERVAL} \
    --split 100,0,0 \
    --$DTYPE \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --no-gradient-accumulation-fusion \
    --distributed-backend ${BE} \
    --tokenizer-type Llama2Tokenizer \
    --save checkpoints/${OUTPUT_PREFIX} \
    --load checkpoints/${OUTPUT_PREFIX} \
    --use-checkpoint-opt_param-scheduler \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --data-file-list ${DATA_FILE_LIST} \
    --data-cache-path ${DATA_CACHE_PATH} \
    $ds_args \
    ${LLAMA_ARGS} \
    ${gpt_args[*]} \
    $custom_args \
    |& tee ${OUTPUT_LOG}
    "
    # >> ${OUTPUT_LOG} 2>&1 &
    # |& tee $OUTPUT_DIR/output.log

# --ffn-hidden-size 11008 \
# --vocab-file $VOCAB_FILE \
# --merge-file $MERGE_FILE \
# --lr-decay-iters 320000 \
# --num-workers 0 \
# --eval-iters ${EVAL_ITERS} \
# --eval-interval ${EVAL_INTERVAL} \
# --lr-warmup-iters 5000 \
# --lr-decay-iters 10000 \
# --accumulate-allreduce-grads-in-fp32 \
# --data-impl mmap \

echo "All DeepSpeed(s): $(which -a deepspeed)"
echo "Using $(which deepspeed)"
ds_report

echo "${run_cmd}"

printf "[!! \e[1;31m%s\e[0m] View output at:\n" "NOTE"
printf "\e[1;34m%s\e[0m\n" "${OUTPUT_LOG}"

eval "${run_cmd}"
set +x

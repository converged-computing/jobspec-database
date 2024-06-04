#!/bin/bash --login
#PBS -l walltime=06:00:00
#PBS -A argonne_tpc
#PBS -q prod
#PBS -l select=48
#PBS -l filesystems=eagle:home

function sourceFile() {
    fp="$1"
    echo "source-ing ${fp}"
    if [[ -f "${fp}" ]]; then
        # shellcheck source="${fp}"
        source "${fp}"
    else
        echo "ERROR: UNABLE TO SOURCE ${fp}"
    fi
}

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ---- 0. Navigate into `$PBS_O_WORKDIR` -------------------------------------
cd "${PBS_O_WORKDIR}" || exit
HERE=$(python3 -c 'import os; print(os.getcwd())')
export HERE
# ---- 1. Assert `./pretrain_gpt_alcf.py` exists: -----------------------------
export EXEC="${HERE}/pretrain_gpt_alcf.py"
[ -f "${EXEC}" ] || exit
# ---- 2. `source ./ALCF/helpers_alcf.sh`: ------------------------------------
sourceFile "${HERE}/ALCF/helpers.sh" || exit
# ---- 3. Call fns from `./ALCF/helpers_alcf.sh` ------------------------------
setEnv || exit                      # 1. load `conda` environment
saveDSenv || exit                   # 2. save env vars to `.deepspeed_env`
ezpz || exit                        # 3. determine WORLD_SIZE, etc. from `PBS_*` vars
makeHostfiles || exit               # 4. create `deepspeed` hostfile from `$PBS_NODEFILE`
setParams || exit                   # 5. set command line arguments to pass to `"${EXEC}"`
buildDSconfig || exit               # 6. create `deepspeed_config.json` from runtime params from ^
setOutput || exit                   # 7. specify output directory for {logs, checkpoints, etc.}
setArgs || exit                     # 8. specify additional `deepspeed` arguments
setData "${DATA_FILE_LIST}"|| exit  # 9. specify `DATA_FILE_LIST` for dolma dataset
setDSlauncher "${HERE}" || exit     # 10. set `launcher` args for `deepspeed ${launcher} ${EXEC} ${args}`
printJobInfo || exit                # 11. print job info
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Take custom args
custom_args=" $@"

# Assert `./hostfile_deepspeed` exists
export hfds="${HERE}/hostfile_deepspeed" && [ -f "${hfds}" ] || exit

# source "${HERE}/venvs/polaris/2024-03-14/bin/activate" || exit
# echo "Using $(which python3)"
# --launcher_args='--pmi=pmix'
    # deepspeed --hostfile $hfds --launcher ${LAUNCHER} ${EXEC} \
    # ${launch_cmd} \
run_cmd="
    deepspeed --hostfile $hfds --launcher MPICH ${EXEC} \
    --use-flash-attn-v2 \
    --$DTYPE \
    --num-workers 0 \
    --split 100,0,0 \
    --log-interval 1 \
    --no-bias-gelu-fusion \
    --lr-decay-style cosine \
    --no-bias-dropout-fusion \
    --no-masked-softmax-fusion \
    --tokenizer-type Llama2Tokenizer \
    --no-gradient-accumulation-fusion \
    --accumulate-allreduce-grads-in-fp32 \
    --use-checkpoint-opt_param-scheduler \
    --lr ${LR} \
    --seq-length $SEQ \
    --save ${CKPT_DIR} \
    --load ${CKPT_DIR} \
    --num-layers ${NLAYERS} \
    --hidden-size ${HIDDEN} \
    --train-iters ${TRAIN_ITER} \
    --eval-iters ${EVAL_ITERS} \
    --distributed-backend ${BE} \
    --num-attention-heads ${HEADS} \
    --save-interval ${SAVE_INTERVAL} \
    --eval-interval ${EVAL_INTERVAL} \
    --max-position-embeddings ${SEQ} \
    --micro-batch-size ${MICRO_BATCH} \
    --data-file-list ${DATA_FILE_LIST} \
    --tensor-model-parallel-size ${TP} \
    --global-batch-size ${GLOBAL_BATCH} \
    --pipeline-model-parallel-size ${PP} \
    --num-key-value-heads ${NUM_KV_HEAD} \
    --data-cache-path ${DATA_CACHE_PATH} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    ${LLAMA_ARGS} \
    $ds_args \
    ${gpt_args[*]} \
    $custom_args \
    |& tee ${OUTPUT_LOG}
    "


echo "All DeepSpeed(s): $(which -a deepspeed)"
echo "Using $(which deepspeed)"
ds_report

echo "${run_cmd}"

printf "[!! \e[1;31m%s\e[0m] View output at:\n" "NOTE"
printf "\e[1;34m%s\e[0m\n" "${OUTPUT_LOG}"
# echo "${OUTPUT_LOG}"
eval "${run_cmd}"
set +x

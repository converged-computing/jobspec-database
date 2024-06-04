#!/bin/bash
#------- qsub option -----------
#PBS -A NBBG
#PBS -l elapstim_req=24:00:00
#PBS -T distrib
#PBS -v USE_DEVDAX=pmemkv
#PBS -v NUM_DEVDAX=1
#------- Program execution -----------
set -eu

JOB_RANK=$(echo "$PBS_JOBID" | awk -F: '{print $1}')
if [ "$JOB_RANK" -ne 0 ]; then
  exit 0
fi

# Requires
# - SPACK_ENV_NAME
# - SCRIPT_DIR
# - OUTPUT_DIR

: "${RDBENCH_STEPS:=640}"
: "${RDBENCH_INTERVAL:=64}"
: "${CHFS_MOUNT_DIR:='/scr/chfs'}"
: "${CHFS_CHUNK_SIZE:=$((64 * 2 ** 10))}"
: "${CHFS_NODE_LIST_CACHE_TIMEOUT:=0}"
: "${CHFS_RPC_TIMEOUT_MSEC:=$((30 * 1000))}"
: "${CHFS_LOG_PRIORITY:='notice'}"
: "${CHFS_NTHREADS:=8}"
: "${CHFS_NIOTHREADS:=2}"
: "${CHFS_RDMA_THRESH:=$((32 * 2 ** 10))}"
: "${CHFS_ASYNC_ACCESS:=0}"
: "${CHFS_LOOKUP_LOCAL:=0}"
: "${FI_UNIVERSE_SIZE:=4096}"

spack env activate "${SPACK_ENV_NAME}"

source "$SCRIPT_DIR/common.sh"
exec 1> >(addtimestamp)
exec 2> >(addtimestamp >&2)

JOB_START=$(timestamp)
NNODES=$(wc --lines "${PBS_NODEFILE}" | awk '{print $1}')
JOBID=$(echo "$PBS_JOBID" | cut -d : -f 2)
JOB_OUTPUT_DIR="${OUTPUT_DIR}/${JOB_START}-${JOBID}-${NNODES}"

echo "prepare the output directory: ${JOB_OUTPUT_DIR}"
mkdir -p "${JOB_OUTPUT_DIR}"
cp "$0" "${JOB_OUTPUT_DIR}"
cp "${PBS_NODEFILE}" "${JOB_OUTPUT_DIR}"
printenv >"${JOB_OUTPUT_DIR}/env.txt"

# romio_hints
ROMIO_HINTS="${JOB_OUTPUT_DIR}/romio_hints"
cp "${SCRIPT_DIR}/romio_hints/off" "${ROMIO_HINTS}"

cmd_dropcaches=(
  mpirun
  -f "$PBS_NODEFILE"
  -np "$NNODES"
  -ppn 1
  dropcaches 3
)

save_job_params() {
  cat <<EOS >"${JOB_OUTPUT_DIR}"/job_params_${runid}.json
{
  "nnodes": ${NNODES},
  "ppn": ${ppn},
  "np": ${np},
  "jobid": "$JOBID",
  "runid": ${runid},
  "lustre_version": "$(lfs --version | awk '{print $2}')",
  "spack_env_name": "${SPACK_ENV_NAME}",
  "storageSystem": "CHFS",
  "chfs_chunk_size": ${CHFS_CHUNK_SIZE},
  "chfs_log_priority": "${CHFS_LOG_PRIORITY}",
  "chfs_nthreads": ${CHFS_NTHREADS},
  "chfs_niothreads": ${CHFS_NIOTHREADS},
  "chfs_rdma_thresh": ${CHFS_RDMA_THRESH},
  "chfs_async_access": ${CHFS_ASYNC_ACCESS},
  "chfs_lookup_local": ${CHFS_LOOKUP_LOCAL},
  "chfs_version": "$(spack find --json chfs | jq -r .[0].version)",
  "fi_universe_size": ${FI_UNIVERSE_SIZE}
}
EOS
}

# start chfs
echo "clean up chfsd and chfuse of prev job"
chfsctl -h "${PBS_NODEFILE}" -m "$CHFS_MOUNT_DIR" kill

args_chfsd=(
  -H 0
  -t "${CHFS_RPC_TIMEOUT_MSEC}"
  -L "${CHFS_LOG_PRIORITY}"
  -T "${CHFS_NTHREADS}"
  -l "${CHFS_NIOTHREADS}"
)

cmd_chfsctl=(
  chfsctl
  -h "${PBS_NODEFILE}"
  -L "${JOB_OUTPUT_DIR}/chfsd_log"
  -p verbs
  -x FI_UNIVERSE_SIZE="${FI_UNIVERSE_SIZE}"
  -i 1
  -NUMACTL "--physcpubind 40-47"
  -D -c /dev/dax0.0
  -O "${args_chfsd[*]}"
  start
)

echo "${cmd_chfsctl[@]}"
chfs_start="$("${cmd_chfsctl[@]}")"
echo "$chfs_start"
eval "$chfs_start"

echo "wait for chfsd to start"
sleep 5

chlist \
  > >(tee "${JOB_OUTPUT_DIR}/chlist.o.txt") \
  2> >(tee "${JOB_OUTPUT_DIR}/chlist.e.txt" >&2)

ppn_list=(
  # 36
  48
)

# workflow_id=0
runid=0
for ppn in "${ppn_list[@]}"; do
  np=$((NNODES * ppn))
  rdbench_length_per_node=$((48 * 2 ** 10)) # 48k * 48k * sizeof(double) = 18 GiB/node
  rdbench_length=$((rdbench_length_per_node * $(sqrt "$NNODES")))

  cmd_mpirun=(
    mpirun
    -f "$PBS_NODEFILE"
    -np "$np"
    -ppn "$ppn"
    -bind-to core
    -genv PATH="${PATH}"
    -genv FI_PROVIDER="verbs;ofi_rxm"
    -genv FI_UNIVERSE_SIZE="${FI_UNIVERSE_SIZE}"
    -genv ROMIO_HINTS="${ROMIO_HINTS}"
    -genv ROMIO_FSTYPE_FORCE=chfs:
    -genv CHFS_SERVER="${CHFS_SERVER:-}"
    -genv CHFS_RPC_TIMEOUT_MSEC="${CHFS_RPC_TIMEOUT_MSEC}"
    -genv CHFS_NODE_LIST_CACHE_TIMEOUT="${CHFS_NODE_LIST_CACHE_TIMEOUT}"
    -genv CHFS_CHUNK_SIZE="${CHFS_CHUNK_SIZE}"
    -genv CHFS_RDMA_THRESH="${CHFS_RDMA_THRESH}"
    -genv CHFS_ASYNC_ACCESS="${CHFS_ASYNC_ACCESS}"
    -genv CHFS_LOOKUP_LOCAL="${CHFS_LOOKUP_LOCAL}"
  )

  cmd_rdbench=(
    rdbench
    --length "$rdbench_length"
    --output "o_"
    --nomkdir
    --iotype view
    --steps "$RDBENCH_STEPS"
    --interval "$RDBENCH_INTERVAL"
    --novalidate
    --disable-initial-output
    --prettify
    --xnp $((6 * $(sqrt "$NNODES")))
    --nosync
  )

  cmd_benchmark=(
    time_json -o "${JOB_OUTPUT_DIR}/time_${runid}.json"
    "${cmd_mpirun[@]}"
    "${cmd_rdbench[@]}"
  )

  save_job_params

  # dropcaches
  echo "${cmd_dropcaches[@]}"
  "${cmd_dropcaches[@]}"

  # rdbench
  echo "${cmd_benchmark[@]}"
  "${cmd_benchmark[@]}" \
    > >(tee "${JOB_OUTPUT_DIR}/rdbench_stdout_${runid}.json") \
    2> >(tee "${JOB_OUTPUT_DIR}/rdbench_stderr_${runid}.txt" >&2)

  runid=$((runid + 1))
done

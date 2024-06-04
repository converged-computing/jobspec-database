#!/bin/bash
#PBS -A NBBG
#PBS -l elapstim_req=24:00:00
#PBS -T distrib
#PBS -v USE_DEVDAX=pmemkv
#PBS -v NUM_DEVDAX=1
set -eu

JOB_RANK=$(echo "$PBS_JOBID" | awk -F: '{print $1}')
if [ "$JOB_RANK" -ne 0 ]; then
  exit 0
fi

# Requires
# - SPACK_ENV_NAME
# - SCRIPT_DIR
# - OUTPUT_DIR

# : "${CHFS_MOUNT_DIR:=/scr/chfs}"
: "${CHFS_CHUNK_SIZE:=$((512 * 2 ** 10))}"
: "${CHFS_NODE_LIST_CACHE_TIMEOUT:=0}"
: "${CHFS_RPC_TIMEOUT_MSEC:=$((30 * 1000))}"
: "${CHFS_LOG_PRIORITY:='notice'}"
: "${CHFS_NTHREADS:=8}"
: "${CHFS_NIOTHREADS:=2}"
: "${CHFS_RDMA_THRESH:=$((32 * 2 ** 10))}"
: "${CHFS_ASYNC_ACCESS:=0}"
: "${CHFS_LOOKUP_LOCAL:=1}"
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
mkdir -p "${JOB_OUTPUT_DIR}/storage/write"
mkdir -p "${JOB_OUTPUT_DIR}/storage/read"
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
# chfsctl -h "${PBS_NODEFILE}" -m "$CHFS_MOUNT_DIR" kill
chfsctl -h "${PBS_NODEFILE}" kill
# rm -rf "${CHFS_MOUNT_DIR:?}/"*

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
  # -m "$CHFS_MOUNT_DIR"
  start
)

# mkdir -p "$CHFS_MOUNT_DIR"
echo "${cmd_chfsctl[@]}"
chfs_start="$("${cmd_chfsctl[@]}")"
echo "$chfs_start"
eval "$chfs_start"

echo "wait for chfsd to start"
sleep 5

chlist \
  > >(tee "${JOB_OUTPUT_DIR}/chlist.o.txt") \
  2> >(tee "${JOB_OUTPUT_DIR}/chlist.e.txt" >&2)

workflow_id=0
runid=0
ppn=32
np=$((NNODES * ppn))

cmd_mpirun=(
  mpirun
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
  -f "$PBS_NODEFILE"
  -np "$np"
  -ppn "$ppn"
  -bind-to core
)

write_cfg="${JOB_OUTPUT_DIR}/write.cfg"
read_cfg="${JOB_OUTPUT_DIR}/read.cfg"
rw_h5="/chfs/rw.h5"

cmd_h5bench_write=(
  "${cmd_mpirun[@]}"
  h5bench_write
  "${write_cfg}"
  "${rw_h5}"
)

cmd_h5bench_read=(
  "${cmd_mpirun[@]}"
  h5bench_read
  "${read_cfg}"
  "${rw_h5}"
)

# generate config files
export JOB_OUTPUT_DIR
envsubst <"${SCRIPT_DIR}/config/write.template.cfg" > "${write_cfg}"
envsubst <"${SCRIPT_DIR}/config/read.template.cfg" > "${read_cfg}"

# run h5bench_write
save_job_params

echo "${cmd_dropcaches[@]}"
"${cmd_dropcaches[@]}"

echo "${cmd_h5bench_write[@]}"
"${cmd_h5bench_write[@]}"

runid=$((runid + 1))

# run h5bench_read
save_job_params

echo "${cmd_dropcaches[@]}"
"${cmd_dropcaches[@]}"

echo "${cmd_h5bench_read[@]}"
"${cmd_h5bench_read[@]}"

runid=$((runid + 1))
workflow_id=$((workflow_id + 1))

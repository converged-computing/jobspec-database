#!/bin/bash
#------- qsub option -----------
#PBS -A NBBG
#PBS -l elapstim_req=24:00:00
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=4.1.5/gcc11.4.0-cuda12.1.0
#------- Program execution -----------
set -eu

# Requires
# - SPACK_ENV_NAME
# - SCRIPT_DIR
# - OUTPUT_DIR
# - BACKEND_DIR

: "${UNIFYFS_DAEMONIZE:=on}"
: "${UNIFYFS_MOUNTPOINT:=/unifyfs}"
: "${UNIFYFS_LOG_VERBOSITY:=3}"
: "${UNIFYFS_LOGIO_SPILL_DIR:=/pmem}"
: "${UNIFYFS_LOGIO_SPILL_SIZE:=$((2 ** 40 + 512 * 2 ** 30))}"
: "${UNIFYFS_LOGIO_SHMEM_SIZE:=0}"
: "${UNIFYFS_LOGIO_CHUNK_SIZE:=$((64 * 2 ** 10))}"
# : "${UNIFYFS_LOGIO_CHUNK_SIZE:=$((1 * 2 ** 20))}"
: "${UNIFYFS_MARGO_TCP:=off}"
: "${UNIFYFS_MARGO_CLIENT_TIMEOUT:=$((60 * 1000))}"
: "${UNIFYFS_MARGO_SERVER_TIMEOUT:=$((60 * 1000))}"
: "${UNIFYFS_MARGO_SERVER_POOL_SIZE:=8}"
: "${UNIFYFS_MARGO_CLIENT_POOL_SIZE:=8}"
: "${UNIFYFS_RUNSTATE_DIR:=/pmem}"
: "${UNIFYFS_SERVER_MAX_APP_CLIENTS:=4096}"
: "${UNIFYFS_SERVER_INIT_TIMEOUT:=120}"
: "${UNIFYFS_SERVER_LOCAL_EXTENTS:=on}"
: "${UNIFYFS_CLIENT_LOCAL_EXTENTS:=on}"
: "${UNIFYFS_CLIENT_NODE_LOCAL_EXTENTS:=on}"
: "${FI_UNIVERSE_SIZE:=4096}"

: "${RDBENCH_STEPS:=640}"
: "${RDBENCH_INTERVAL:=64}"

export UNIFYFS_DAEMONIZE
export UNIFYFS_MOUNTPOINT
export UNIFYFS_LOG_DIR
export UNIFYFS_LOG_VERBOSITY
export UNIFYFS_LOGIO_SPILL_DIR
export UNIFYFS_LOGIO_SPILL_SIZE
export UNIFYFS_LOGIO_SHMEM_SIZE
export UNIFYFS_LOGIO_CHUNK_SIZE
export UNIFYFS_MARGO_TCP
export UNIFYFS_MARGO_CLIENT_TIMEOUT
export UNIFYFS_MARGO_SERVER_TIMEOUT
export UNIFYFS_MARGO_SERVER_POOL_SIZE
export UNIFYFS_MARGO_CLIENT_POOL_SIZE
export UNIFYFS_RUNSTATE_DIR
export UNIFYFS_SERVER_MAX_APP_CLIENTS
export UNIFYFS_SERVER_INIT_TIMEOUT
export UNIFYFS_SERVER_LOCAL_EXTENTS
export UNIFYFS_CLIENT_LOCAL_EXTENTS
export UNIFYFS_CLIENT_NODE_LOCAL_EXTENTS

spack env activate "${SPACK_ENV_NAME}"
module purge
module load "openmpi/$NQSV_MPI_VER"

source "$SCRIPT_DIR/common.sh"
exec 1> >(addtimestamp)
exec 2> >(addtimestamp >&2)

JOB_START=$(timestamp)
NNODES=$(wc --lines "${PBS_NODEFILE}" | awk '{print $1}')
JOBID=$(echo "$PBS_JOBID" | cut -d : -f 2)
JOB_OUTPUT_DIR="${OUTPUT_DIR}/${JOB_START}-${JOBID}-${NNODES}"
JOB_BACKEND_DIR="${BACKEND_DIR}/$(basename -- "${JOB_OUTPUT_DIR}")"
UNIFYFS_LOG_DIR="${JOB_OUTPUT_DIR}/unifyfsd_log"
UNIFYFS_SERVER_HOSTFILE="${JOB_OUTPUT_DIR}/unifyfsd.hosts"
UNIFYFS_PRELOAD_LIB="$(spack location -i unifyfs)/lib/libunifyfs_mpi_gotcha.so"
UNIFYFS_SHAREDFS_DIR="${JOB_BACKEND_DIR}"
RANKFILE="${JOB_OUTPUT_DIR}/rankfile"

IFS=" " read -r -a nqsii_mpiopts_array <<<"$NQSII_MPIOPTS"

echo "prepare the output directory: ${JOB_OUTPUT_DIR}"
mkdir -p "${JOB_OUTPUT_DIR}"
mkdir -p "${UNIFYFS_LOG_DIR}"
cp "$0" "${JOB_OUTPUT_DIR}"
cp "${PBS_NODEFILE}" "${JOB_OUTPUT_DIR}"
printenv >"${JOB_OUTPUT_DIR}/env.txt"

# romio_hints
ROMIO_HINTS="${JOB_OUTPUT_DIR}/romio_hints"
cp "${SCRIPT_DIR}/romio_hints/off" "${ROMIO_HINTS}"

echo "prepare backend dir: ${JOB_BACKEND_DIR}"
mkdir -p "${JOB_BACKEND_DIR}"
trap 'rm -rf "${JOB_BACKEND_DIR}" ; exit 1' 1 2 3 15
trap 'rm -rf "${JOB_BACKEND_DIR}" ; exit 0' EXIT

ppn_list=(
  36
  # 48
)

cmd_dropcaches=(
  mpirun
  "${nqsii_mpiopts_array[@]}"
  -np "$NNODES"
  -map-by ppr:1:node
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
  "storageSystem": "UnifyFS",
  "fi_universe_size": ${FI_UNIVERSE_SIZE},
  "UNIFYFS_LOG_VERBOSITY": ${UNIFYFS_LOG_VERBOSITY},
  "UNIFYFS_LOGIO_SPILL_DIR": "${UNIFYFS_LOGIO_SPILL_DIR}",
  "UNIFYFS_LOGIO_SPILL_SIZE": ${UNIFYFS_LOGIO_SPILL_SIZE},
  "UNIFYFS_LOGIO_SHMEM_SIZE": ${UNIFYFS_LOGIO_SHMEM_SIZE},
  "UNIFYFS_LOGIO_CHUNK_SIZE": ${UNIFYFS_LOGIO_CHUNK_SIZE},
  "UNIFYFS_MARGO_TCP": "${UNIFYFS_MARGO_TCP}",
  "UNIFYFS_MARGO_CLIENT_TIMEOUT": ${UNIFYFS_MARGO_CLIENT_TIMEOUT},
  "UNIFYFS_MARGO_SERVER_TIMEOUT": ${UNIFYFS_MARGO_SERVER_TIMEOUT},
  "UNIFYFS_MARGO_SERVER_POOL_SIZE": ${UNIFYFS_MARGO_SERVER_POOL_SIZE},
  "UNIFYFS_MARGO_CLIENT_POOL_SIZE": ${UNIFYFS_MARGO_CLIENT_POOL_SIZE},
  "UNIFYFS_RUNSTATE_DIR": "${UNIFYFS_RUNSTATE_DIR}",
  "UNIFYFS_SERVER_MAX_APP_CLIENTS": ${UNIFYFS_SERVER_MAX_APP_CLIENTS},
  "UNIFYFS_SERVER_INIT_TIMEOUT": ${UNIFYFS_SERVER_INIT_TIMEOUT},
  "UNIFYFS_SERVER_LOCAL_EXTENTS": "${UNIFYFS_SERVER_LOCAL_EXTENTS}",
  "UNIFYFS_CLIENT_LOCAL_EXTENTS": "${UNIFYFS_CLIENT_LOCAL_EXTENTS}",
  "UNIFYFS_CLIENT_NODE_LOCAL_EXTENTS": "${UNIFYFS_CLIENT_NODE_LOCAL_EXTENTS}"
}
EOS
}

# start UnifyFS
args_unifyfs_env_vars=()
while IFS='=' read -r key val; do
  if [[ $key == UNIFYFS_* ]]; then
    args_unifyfs_env_vars+=(-x "$key")
  fi
done < <(env)

# create unifyfsd.hosts
echo "$NNODES" >"$UNIFYFS_SERVER_HOSTFILE"
cat "$PBS_NODEFILE" >>"$UNIFYFS_SERVER_HOSTFILE"

# create rankfile
touch "$RANKFILE"
rank=0
while IFS= read -r node; do
  echo "rank $rank=$node slot=40-47" >>"$RANKFILE"
  rank=$((rank + 1))
done <"$PBS_NODEFILE"

cmd_unifyfs_start=(
  mpirun
  "${nqsii_mpiopts_array[@]}"
  -x PATH
  "${args_unifyfs_env_vars[@]}"
  -np "$NNODES"
  -rf "$RANKFILE"
  unifyfsd
  --unifyfs-cleanup on
  --server-hostfile "$UNIFYFS_SERVER_HOSTFILE"
  --sharedfs-dir "$UNIFYFS_SHAREDFS_DIR"
)

echo "${cmd_unifyfs_start[@]}"
"${cmd_unifyfs_start[@]}"

echo "wait for unifyfsd to start"
sleep 30

unifyfs-ls \
  > >(tee "${JOB_OUTPUT_DIR}/unifyfs-ls.o.txt") \
  2> >(tee "${JOB_OUTPUT_DIR}/unifyfs-ls.e.txt" >&2)

workflow_id=0
runid=0
for ppn in "${ppn_list[@]}"; do
  np=$((NNODES * ppn))
  rdbench_length_per_node=$((48 * 2 ** 10)) # 48k * 48k * sizeof(double) = 18 GiB/node
  rdbench_length=$((rdbench_length_per_node * $(sqrt "$NNODES")))

  cmd_mpirun=(
    mpirun
    "${nqsii_mpiopts_array[@]}"
    -np "$np"
    -map-by "ppr:${ppn}:node"
    -bind-to core
    -mca io romio321
    -x PATH
    -x LD_PRELOAD="${UNIFYFS_PRELOAD_LIB}"
    "${args_unifyfs_env_vars[@]}"
    # -x FI_PROVIDER="verbs;ofi_rxm"
    -x FI_UNIVERSE_SIZE="${FI_UNIVERSE_SIZE}"
    -x ROMIO_HINTS="${ROMIO_HINTS}"
  )

  cmd_rdbench=(
    rdbench
    --length "$rdbench_length"
    --output "${UNIFYFS_MOUNTPOINT}/o_"
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
  workflow_id=$((workflow_id + 1))
done

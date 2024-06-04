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
: "${UNIFYFS_LOGIO_SPILL_SIZE:=$((2**40+512*2**30))}"
: "${UNIFYFS_LOGIO_SHMEM_SIZE:=0}"
: "${UNIFYFS_LOGIO_CHUNK_SIZE:=$((1*2**20))}"
: "${UNIFYFS_MARGO_TCP:=off}"
: "${UNIFYFS_MARGO_CLIENT_TIMEOUT:=$((60*1000))}"
: "${UNIFYFS_MARGO_SERVER_TIMEOUT:=$((60*1000))}"
: "${UNIFYFS_MARGO_SERVER_POOL_SIZE:=8}"
: "${UNIFYFS_MARGO_CLIENT_POOL_SIZE:=8}"
: "${UNIFYFS_RUNSTATE_DIR:=/pmem}"
: "${UNIFYFS_SERVER_MAX_APP_CLIENTS:=4096}"
: "${UNIFYFS_SERVER_INIT_TIMEOUT:=120}"
: "${UNIFYFS_SERVER_LOCAL_EXTENTS:=on}"
: "${UNIFYFS_CLIENT_LOCAL_EXTENTS:=on}"
: "${UNIFYFS_CLIENT_NODE_LOCAL_EXTENTS:=on}"
: "${FI_UNIVERSE_SIZE:=4096}"

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
  32
)

min_io_size_per_proc=$((4 * 2 ** 30)) # 4 GiB/proc * 16 ppn == 64 GiB/node

xfer_size_list=(
  1M
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
  "UNIFYFS_MARGO_TCP": ${UNIFYFS_MARGO_TCP},
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
while IFS='=' read -r key val ; do
  if [[ $key == UNIFYFS_* ]]; then
    args_unifyfs_env_vars+=(-x "$key")
  fi
done < <(env)

# create unifyfsd.hosts
echo "$NNODES" > "$UNIFYFS_SERVER_HOSTFILE"
cat "$PBS_NODEFILE" >> "$UNIFYFS_SERVER_HOSTFILE"

# create rankfile
touch "$RANKFILE"
rank=0
while IFS= read -r node; do
  echo "rank $rank=$node slot=40-47" >> "$RANKFILE"
  rank=$((rank+1))
done < "$PBS_NODEFILE"

unifyfs_env_vars=()
while IFS='=' read -r key val ; do
  if [[ $key == UNIFYFS_* ]]; then
    unifyfs_env_vars+=(-x "$key")
  fi
done < <(env)

cmd_unifyfs_start=(
  mpirun
  "${nqsii_mpiopts_array[@]}"
  -x PATH
  "${unifyfs_env_vars[@]}"
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

  for xfer_size_human in "${xfer_size_list[@]}"; do
    # SSF-segmented
    xfer_size=$(numfmt --from=iec "$xfer_size_human")
    segment_count=1
    block_size=$(((min_io_size_per_proc + segment_count - 1) / segment_count))
    block_size=$(((block_size + xfer_size - 1) / xfer_size * xfer_size))

    test_file="/unifyfs/testfile${workflow_id}"

    cmd_mpirun=(
      mpirun
      "${nqsii_mpiopts_array[@]}"
      -np "$np"
      -map-by "ppr:${ppn}:node"
      -bind-to core
      -mca io romio321
      -x PATH
      -x LD_PRELOAD="${UNIFYFS_PRELOAD_LIB}"
      "${unifyfs_env_vars[@]}"
      # -x FI_PROVIDER="verbs;ofi_rxm"
      -x FI_UNIVERSE_SIZE="${FI_UNIVERSE_SIZE}"
      -x ROMIO_HINTS="${ROMIO_HINTS}"
    )

    cmd_ior=(
      ior
      -a MPIIO
      -l timestamp   # --data
      -g             # intraTestBarriers – use barriers between open, write/read, and close
      -G -1401473791 # setTimeStampSignature – set value for time stamp signature
      -k             # keepFile – don’t remove the test file(s) on program exit
      -e             # fsync
      -i 1
      -s "$segment_count"
      -b "$block_size"
      -t "$xfer_size"
      -o "$test_file"
      -O "summaryFormat=JSON"
    )

    save_job_params

    # dropcaches
    echo "${cmd_dropcaches[@]}"
    "${cmd_dropcaches[@]}"

    # write
    cmd_write=(
      time_json -o "${JOB_OUTPUT_DIR}/time_${runid}.json"
      "${cmd_mpirun[@]}"
      "${cmd_ior[@]}"
      -O "summaryFile=${JOB_OUTPUT_DIR}/ior_summary_${runid}.json"
      -w
    )

    echo "${cmd_write[@]}"
    "${cmd_write[@]}" \
      >"${JOB_OUTPUT_DIR}/ior_stdout_${runid}.txt" \
      2>"${JOB_OUTPUT_DIR}/ior_stderr_${runid}.txt"

    runid=$((runid + 1))

    save_job_params

    # dropcaches
    echo "${cmd_dropcaches[@]}"
    "${cmd_dropcaches[@]}"

    # read remote
    cmd_read_remote=(
      time_json -o "${JOB_OUTPUT_DIR}/time_${runid}.json"
      "${cmd_mpirun[@]}"
      "${cmd_ior[@]}"
      -O "summaryFile=${JOB_OUTPUT_DIR}/ior_summary_${runid}.json"
      -r
      -C
      -Q 1
    )

    echo "${cmd_read_remote[@]}"
    "${cmd_read_remote[@]}" \
      >"${JOB_OUTPUT_DIR}/ior_stdout_${runid}.txt" \
      2>"${JOB_OUTPUT_DIR}/ior_stderr_${runid}.txt"

    runid=$((runid + 1))

    save_job_params

    # dropcaches
    echo "${cmd_dropcaches[@]}"
    "${cmd_dropcaches[@]}"

    # read local
    cmd_read_local=(
      time_json -o "${JOB_OUTPUT_DIR}/time_${runid}.json"
      "${cmd_mpirun[@]}"
      "${cmd_ior[@]}"
      -O "summaryFile=${JOB_OUTPUT_DIR}/ior_summary_${runid}.json"
      -r
    )

    echo "${cmd_read_local[@]}"
    "${cmd_read_local[@]}" \
      >"${JOB_OUTPUT_DIR}/ior_stdout_${runid}.txt" \
      2>"${JOB_OUTPUT_DIR}/ior_stderr_${runid}.txt"

    runid=$((runid + 1))
    workflow_id=$((workflow_id + 1))
  done
done

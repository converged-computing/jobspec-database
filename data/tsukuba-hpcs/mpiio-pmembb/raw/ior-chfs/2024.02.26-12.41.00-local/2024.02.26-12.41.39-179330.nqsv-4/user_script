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

: "${CHFS_MOUNT_DIR:='/scr/chfs'}"
: "${CHFS_CHUNK_SIZE:=$((512 * 2 ** 10))}"
: "${CHFS_NODE_LIST_CACHE_TIMEOUT:=0}"
: "${CHFS_RPC_TIMEOUT_MSEC:=$((30*1000))}"
: "${CHFS_LOG_PRIORITY:='notice'}"
: "${CHFS_NTHREADS:=8}"
: "${CHFS_NIOTHREADS:=2}"
: "${CHFS_RDMA_THRESH:=$((32*2**10))}"
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
cp "$0" "${JOB_OUTPUT_DIR}"
cp "${PBS_NODEFILE}" "${JOB_OUTPUT_DIR}"
printenv >"${JOB_OUTPUT_DIR}/env.txt"

# romio_hints
ROMIO_HINTS="${JOB_OUTPUT_DIR}/romio_hints"
cp "${SCRIPT_DIR}/romio_hints/off" "${ROMIO_HINTS}"

ppn_list=(
  # 8
  32
)

min_io_size_per_proc=$((20 * 2 ** 30)) # 20 GiB/proc, 20 * 32 ppn == 640 GiB/node
# min_io_size_per_proc=$((8 * 2 ** 30)) # 8 GiB/proc, 8 * 32 ppn == 256 GiB/node

xfer_size_list=(
  512K
)

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

workflow_id=0
runid=0
for ppn in "${ppn_list[@]}"; do
  np=$((NNODES * ppn))

  for xfer_size_human in "${xfer_size_list[@]}"; do
    # SSF-strided
    xfer_size=$(numfmt --from=iec "$xfer_size_human")
    block_size=$xfer_size
    segment_count=$((min_io_size_per_proc / xfer_size))

    test_file="chfs_test_file"

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

    # rm test_file
    rm "${test_file}"

    runid=$((runid + 1))
    workflow_id=$((workflow_id + 1))
  done
done

echo "stop chfsd"
chfsctl -h "${PBS_NODEFILE}" kill

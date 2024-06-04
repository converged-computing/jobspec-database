#!/bin/bash
#------- qsub option -----------
#PBS -A NBBG
#PBS -l elapstim_req=24:00:00
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=${NQSV_MPI_VER}
#PBS -v USE_PMEM_BEEOND=1
#------- Program execution -----------
set -eu

module load use.own
module load "openmpi/$NQSV_MPI_VER"

# Requires
# - SPACK_ENV_NAME
# - SCRIPT_DIR
# - OUTPUT_DIR
# - BACKEND_DIR

spack env activate "${SPACK_ENV_NAME}"

source "$SCRIPT_DIR/common.sh"
exec 1> >(addtimestamp)
exec 2> >(addtimestamp >&2)

JOB_START=$(timestamp)
NNODES=$(wc --lines "${PBS_NODEFILE}" | awk '{print $1}')
JOBID=$(echo "$PBS_JOBID" | cut -d : -f 2)
JOB_OUTPUT_DIR="${OUTPUT_DIR}/${JOB_START}-${JOBID}-${NNODES}"
JOB_BACKEND_DIR="${BACKEND_DIR}/$(basename -- "${JOB_OUTPUT_DIR}")"

IFS=" " read -r -a nqsii_mpiopts_array <<<"$NQSII_MPIOPTS"

echo "prepare the output directory: ${JOB_OUTPUT_DIR}"
mkdir -p "${JOB_OUTPUT_DIR}"
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
  # 48 32 16 8 4 2 1
  # 48
  16
)

min_io_size_per_proc=$((20 * 2 ** 30)) # 20 GiB/proc, 20 * 16 ppn == 320 GiB/node
# min_io_size_per_proc=$((8 * 2 ** 30)) # 8 GiB/proc, 8 * 16 ppn == 128 GiB/node
# min_io_size_per_proc=$((4 * 2 ** 30)) # 4 GiB/proc, 4 * 16 ppn == 64 GiB/node

segment_count_list=(
  1
  # 128
)

xfer_size_list=(
  2M
  # 1M
  # 512K 256K
  # 128K 64K 32K 16K 8K 4K 2K 1K
  # 512 256
  # 47008 3901
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
  "storageSystem": "BeeOND"
}
EOS
}

workflow_id=0
runid=0
for ppn in "${ppn_list[@]}"; do
  np=$((NNODES * ppn))

  for segment_count in "${segment_count_list[@]}"; do
    for xfer_size_human in "${xfer_size_list[@]}"; do
      xfer_size=$(numfmt --from=iec "$xfer_size_human")
      block_size=$(((min_io_size_per_proc + segment_count - 1) / segment_count))
      block_size=$(((block_size + xfer_size - 1) / xfer_size * xfer_size))
      block_size=$(((block_size + 7) / 8 * 8))

      echo "prepare test_dir"
      test_dir="/beeond/${workflow_id}"
      test_file="${test_dir}/test_file"
      mkdir -p "${test_dir}"
      cmd_beegfs_ctl_set=(
        /sbin/beegfs-ctl
        --setpattern
        --mount=/beeond
        "--chunksize=${xfer_size}"
        "--numtargets=${NNODES}"
        "${test_dir}"
      )
      cmd_beegfs_ctl_get=(
        /sbin/beegfs-ctl
        --getentryinfo
        --mount=/beeond
        "${test_dir}"
      )

      "${cmd_beegfs_ctl_set[@]}"
      "${cmd_beegfs_ctl_get[@]}" \
        > "${JOB_OUTPUT_DIR}/beegfs_getentryinfo.txt"

      cmd_mpirun=(
        mpirun
        "${nqsii_mpiopts_array[@]}"
        -x PATH
        -x ROMIO_FSTYPE_FORCE=ufs:
        -x ROMIO_HINTS="${ROMIO_HINTS}"
        -mca hook_pmembb_enable false
        -mca io romio341
        -mca osc ucx
        -mca pml ucx
        -mca osc_ucx_acc_single_intrinsic true
        -np "$np"
        -map-by "ppr:${ppn}:node"
      )

      cmd_ior=(
        ior
        -a MPIIO
        # --posix.odirect
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
      if ! "${cmd_write[@]}" \
         > "${JOB_OUTPUT_DIR}/ior_stdout_${runid}.txt" \
         2> "${JOB_OUTPUT_DIR}/ior_stderr_${runid}.txt"; then
        find /pmem/log -type f -name "*.log" -exec tail {} +
        mkdir -p "${JOB_OUTPUT_DIR}/pmem_log"
        cp -R /pmem/log "${JOB_OUTPUT_DIR}/pmem_log"
        exit
      fi

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
        > "${JOB_OUTPUT_DIR}/ior_stdout_${runid}.txt" \
        2> "${JOB_OUTPUT_DIR}/ior_stderr_${runid}.txt"

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
        > "${JOB_OUTPUT_DIR}/ior_stdout_${runid}.txt" \
        2> "${JOB_OUTPUT_DIR}/ior_stderr_${runid}.txt"


      # rm test_file
      rm "${test_file}"

      runid=$((runid + 1))
      workflow_id=$((workflow_id + 1))
    done
  done
done

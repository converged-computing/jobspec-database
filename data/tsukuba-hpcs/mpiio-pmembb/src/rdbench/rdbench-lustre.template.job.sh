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

: ${RDBENCH_STEPS:=640}
: ${RDBENCH_INTERVAL:=64}

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

save_job_params() {
  cat <<EOS >"${JOB_OUTPUT_DIR}"/job_params_${runid}.json
{
  "nnodes": ${NNODES},
  "ppn": ${ppn},
  "np": ${np},
  "jobid": "$JOBID",
  "runid": ${runid},
  "lustre_version": "$(lfs --version | awk '{print $2}')",
  "lustre_stripe_size": ${stripe_size},
  "lustre_stripe_count": ${stripe_count},
  "spack_env_name": "${SPACK_ENV_NAME}",
  "storageSystem": "Lustre"
}
EOS
}

ppn_list=(48 36)


# workflow_id=0
runid=0
for ppn in "${ppn_list[@]}"; do
  np=$((NNODES * ppn))
  rdbench_length_per_node=$((48 * 2 ** 10)) # 48k * 48k * sizeof(double) = 18 GiB/node
  rdbench_length=$((rdbench_length_per_node * $(sqrt "$NNODES")))

  max_stripe_count=2000
  stripe_size=$((rdbench_length_per_node * 8 / 6)) # sizeof(double)=8, xnp_per_node=6
  stripe_count=$np
  if [ $stripe_count -gt $max_stripe_count ]; then
    stripe_count=$max_stripe_count
  fi

  echo "prepare test_dir"
  TEST_DIR="${JOB_BACKEND_DIR}/${runid}"
  mkdir -p "${TEST_DIR}"
  cmd_lfs_setstripe=(
    lfs setstripe
    -C "$stripe_count"
    --stripe-index -1
    --stripe-size "${stripe_size}"
    "${TEST_DIR}"
  )
  "${cmd_lfs_setstripe[@]}"
  lfs getstripe "${TEST_DIR}"

  cmd_mpirun=(
    mpirun
    "${nqsii_mpiopts_array[@]}"
    -x PATH
    -x ROMIO_FSTYPE_FORCE=ufs:
    -x ROMIO_HINTS="${ROMIO_HINTS}"
    -mca hook_pmembb_enable false
    -mca io romio341
    -mca osc ucx
    -mca osc_ucx_acc_single_intrinsic true
    -np "$np"
    -map-by "ppr:${ppn}:node"
  )

  cmd_rdbench=(
    rdbench
    --length "$rdbench_length"
    --output "/${TEST_DIR}/o_"
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

  cmd_dropcaches=(
    mpirun
    "${nqsii_mpiopts_array[@]}"
    -np "$NNODES"
    -map-by ppr:1:node
    dropcaches 3
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

  # dump test dir
  tree -s /beeond >"${JOB_OUTPUT_DIR}/out_tree_${ppn}"

  # rm test dir
  rm -rf "${TEST_DIR}"

  runid=$((runid + 1))
done

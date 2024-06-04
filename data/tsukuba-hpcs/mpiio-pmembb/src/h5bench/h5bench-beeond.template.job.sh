#!/bin/bash
#PBS -A NBBG
#PBS -l elapstim_req=24:00:00
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=${NQSV_MPI_VER}
#PBS -v USE_PMEM_BEEOND=1
set -eu

module load use.own
module load "openmpi/$NQSV_MPI_VER"

# Requires
# - SPACK_ENV_NAME
# - SCRIPT_DIR
# - OUTPUT_DIR

spack env activate "${SPACK_ENV_NAME}"

source "$SCRIPT_DIR/common.sh"
exec 1> >(addtimestamp)
exec 2> >(addtimestamp >&2)

JOB_START=$(timestamp)
NNODES=$(wc --lines "${PBS_NODEFILE}" | awk '{print $1}')
JOBID=$(echo "$PBS_JOBID" | cut -d : -f 2)
JOB_OUTPUT_DIR="${OUTPUT_DIR}/${JOB_START}-${JOBID}-${NNODES}"

IFS=" " read -r -a nqsii_mpiopts_array <<<"$NQSII_MPIOPTS"

echo "prepare the output directory: ${JOB_OUTPUT_DIR}"
mkdir -p "${JOB_OUTPUT_DIR}"
cp "$0" "${JOB_OUTPUT_DIR}"
cp "${PBS_NODEFILE}" "${JOB_OUTPUT_DIR}"
printenv >"${JOB_OUTPUT_DIR}/env.txt"
cd "${JOB_OUTPUT_DIR}"

# romio_hints
ROMIO_HINTS="${JOB_OUTPUT_DIR}/romio_hints"
cp "${SCRIPT_DIR}/romio_hints/off" "${ROMIO_HINTS}"

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
  "storageSystem": "BeeOND",
  "beeond_chunksize": ${beeond_chunk_size}
}
EOS
}

workflow_id=0
runid=0
ppn=32

np=$((NNODES * ppn))
beeond_chunk_size=$((512*2**10))

echo "prepare test_dir"
TEST_DIR="/beeond/${workflow_id}"
mkdir -p "${TEST_DIR}"

cmd_beegfs_ctl_set=(
  /sbin/beegfs-ctl
  --setpattern
  --mount=/beeond
  "--chunksize=${beeond_chunk_size}"
  "--numtargets=${NNODES}"
  "${TEST_DIR}"
)
cmd_beegfs_ctl_get=(
  /sbin/beegfs-ctl
  --getentryinfo
  --mount=/beeond
  "${TEST_DIR}"
)

"${cmd_beegfs_ctl_set[@]}"
"${cmd_beegfs_ctl_get[@]}" \
  >"${JOB_OUTPUT_DIR}/beegfs_getentryinfo_${workflow_id}.txt"

args_mpirun_common=(
  "${nqsii_mpiopts_array[@]}"
  -x PATH
  -x ROMIO_FSTYPE_FORCE=ufs:
  -x ROMIO_HINTS="${ROMIO_HINTS}"
  -mca hook_pmembb_enable false
  -mca io romio341
  -mca osc ucx
  -mca pml ucx
  -mca osc_ucx_acc_single_intrinsic true
)
args_mpirun_write=(
  "${args_mpirun_common[@]}"
  -np "$np"
  -map-by "ppr:${ppn}:node"
)
args_mpirun_read=(
  "${args_mpirun_common[@]}"
  -np "$np"
  -map-by "ppr:${ppn}:node"
)

wf_write="${JOB_OUTPUT_DIR}/write.json"
wf_read="${JOB_OUTPUT_DIR}/read.json"

export MPI_ARGS
export TEST_DIR
MPI_ARGS="${args_mpirun_write[*]}"
envsubst <"${SCRIPT_DIR}/workflows/write.json" >"${wf_write}"
MPI_ARGS="${args_mpirun_read[*]}"
envsubst <"${SCRIPT_DIR}/workflows/read.json" >"${wf_read}"

save_job_params
# dropcaches
echo "${cmd_dropcaches[@]}"
"${cmd_dropcaches[@]}"

time_json -o "${JOB_OUTPUT_DIR}/time_${runid}.json" \
  h5bench -d "$wf_write"
rsync -av --exclude='*.h5' "$TEST_DIR/" "$JOB_OUTPUT_DIR/"
runid=$((runid + 1))
save_job_params

# dropcaches
echo "${cmd_dropcaches[@]}"
"${cmd_dropcaches[@]}"

time_json -o "${JOB_OUTPUT_DIR}/time_${runid}.json" \
  h5bench -d "$wf_read"
rsync -av --exclude='*.h5' "$TEST_DIR/" "$JOB_OUTPUT_DIR/"
runid=$((runid + 1))

# dump h5 files
tree -s "$TEST_DIR" >"${JOB_OUTPUT_DIR}/tree.txt"
h5ls -r "$TEST_DIR/storage/rw.h5" >"${JOB_OUTPUT_DIR}/h5ls_rw.txt"

runid=$((runid + 1))
workflow_id=$((workflow_id + 1))

#! /usr/bin/env bash
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1G
#SBATCH --array=1-40%1
#SBATCH --output=exp/slurm_logs/slurm-%A.out

# Compute Canada (Alliance Canada) run script wrapper for sbatch. Uses job
# array for dynamic duration and sets some flags according to slurm env
# variables.
#
# e.g.
# sbatch --gres=gpu:t4:1 ./scripts/cc_run.sh -m cpc.small -v 1
# sbatch ./scripts/cc_run.sh ./scripts/zrc_run.sh -m cpc.small -v 1

# we don't rely on pytorch-lightning to requeue b/c CC doesn't allow requeuing
# via scontrol. Following https://docs.alliancecan.ca/wiki/Running_jobs, we
# create an array of jobs instead. If we can complete the run.sh call, we
# cancel all remaining elements in the array.

export NCCL_DEBUG=INFO

source scripts/cc_env.sh

P="${SLURM_NTASKS:-1}"
W="${SLURM_CPUS_PER_TASK:-4}"

# some nodes are buggy and will fail these tests. Don't use em.
pytest tests

script="./run.sh"
if [ -f "$1" ]; then
  script="$1"
  shift
fi

# export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:512'
"$script" -zs -p "$P" -w "$W" -x "--quiet" "$@"
r=$?
sleep 5
scancel --state=PENDING "${SLURM_ARRAY_JOB_ID}"
sleep 10
exit $r

#!/bin/bash
#SBATCH --job-name=moses-singularity
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --partition=longrunning
#SBATCH --chdir=/home/staff/haukurpj/SMT
#SBATCH --time=8:01:00
#SBATCH --output=logs/%x-%j.out

# check if script is started via SLURM or bash
# if with SLURM: there variable '$SLURM_JOB_ID' will exist
if [ -n "$SLURM_JOB_ID" ];  then
    export THREADS="$SLURM_CPUS_PER_TASK"
    export MEMORY="$SLURM_MEM_PER_NODE"
else
    export THREADS=4
    export MEMORY=4096
fi
export MOSESDECODER="/opt/moses"
export MOSESDECODER_TOOLS="/opt/moses_tools"
singularity exec \
  -B "$WORK_DIR":"$WORK_DIR" \
  -B "$REPO_DIR":"$REPO_DIR" \
  -B /data/tools/anaconda:/data/tools/anaconda \
	docker://haukurp/moses-smt:1.1.0 \
  "$@"
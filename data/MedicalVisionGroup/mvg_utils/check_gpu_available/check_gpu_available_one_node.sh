#!/bin/bash
#
#SBATCH --job-name=chk_gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB

set -e # fail fully on first line failure

echo "$(hostname)"

if [ -z "$SLURM_ARRAY_TASK_ID" ]
then
    # Not in Slurm Job Array - running in single mode
    JOB_ID=$SLURM_JOB_ID
else
    # In array
    JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi

srun nvidia-smi

# mv "slurm-${JOB_ID}.out" "$(hostname)".out

# If following error happens, will not reach here.
#     Failed to initialize NVML: Driver/library version mismatch
# So remove output file for slurm job if execution reach here.
rm "slurm-${JOB_ID}.out"

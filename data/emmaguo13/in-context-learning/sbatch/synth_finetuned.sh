#!/bin/bash

#SBATCH --job-name=in-context-learning
#SBATCH --open-mode=append
#SBATCH --output=logs/out/%x_%j.txt
#SBATCH --error=logs/err/%x_%j.txt
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:2
#SBATCH --account=fc_ocow
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --array=1-1
#SBATCH -o icl-%A_%a.out
#SBATCH -e icl-%A_%a.err

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))
PARALLEL_N=4
JOB_N=1

COM_ID_S=$((TASK_ID * PARALLEL_N + 1))

# Check if a config file is provided
if [ -z "$1" ]; then
  echo "Error: No configuration file provided. Usage: $0 <config-file>"
  exit 1
fi

CONFIG_FILE=$1

module load gnu-parallel

declare -a commands=(
 [1]="singularity exec --userns --fakeroot --nv --writable-tmpfs -B /usr/lib64 -B /var/lib/dcv-gl --overlay /global/scratch/users/emmaguo/singularity/overlay-50G-10M.ext3:ro /global/scratch/users/emmaguo/singularity/cuda11.5-cudnn8-devel-ubuntu18.04.sif /bin/bash -c 'source ~/.bashrc && conda activate in-context-learning && cd /global/home/users/emmaguo/in-context-learning/src && MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=online python train.py --config $CONFIG_FILE'"
)

echo "General output file path: logs/out/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.txt"
echo "General error file path: logs/err/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.txt"
echo "Array-specific output file path: icl-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
echo "Array-specific error file path: icl-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

parallel --delay 20 --linebuffer -j 4 {1} ::: "${commands[@]:$COM_ID_S:$PARALLEL_N}"

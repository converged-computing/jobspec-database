#!/usr/bin/env bash
#
#SBATCH -J eval_res50  # give the job a name   
#***SBATCH --partition=batch_default ***
# 
# 1 node, 1 CPU per node (total 1 CPU), wall clock time of hours
#
#SBATCH -N 1                  ## Node count
#SBATCH --ntasks-per-node=1   ## Processors per node
#SBATCH --ntasks=1            ## Tasks
#SBATCH --gres=gpu:1          ## GPUs
#SBATCH --cpus-per-task=8     ## CPUs per task; number of threads of each task
#SBATCH -t 256:00:00          ## Walltime
#SBATCH --mem=40GB
#SBATCH -p research
#SBATCH --exclude=euler[01-09],euler[11-12],euler[14],euler[24-27]
#SBATCH --error=./eulerlog/res50_job_slurm_%A_%a.err
#SBATCH --output=./eulerlog/res50_job_slurm_%A_%a.out
source ~/.bashrc

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

#*** for testing CUDA, run python code below
echo "======== testing CUDA available ========"
echo "running on machine: " $(hostname -s)
python - << EOF
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))
EOF

echo "======== run with different inputs ========"

# python tools/eval_baseline.py \
#     --limit 5000 \
#     --skip_block $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/input_file_skip_block.txt )

python eval_constrain_macs.py \
    --load_path './log/train_resnet50_imagenet' \
    --limit 5000 \
    --skip_block $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/input_file_skip_block.txt )


# sbatch --array=1-16 jobArrayScript.sh

# --dependency=afterany:341497

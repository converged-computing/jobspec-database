#!/usr/bin/env bash
#
#SBATCH -J ada_vit  # give the job a name   
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
#SBATCH -p lianglab,research
#SBATCH --exclude=euler[01-09],euler[11-12],euler[14],euler[24-27]
#SBATCH --error=./eulerlog/ft_array_job_slurm_%A_%a.err
#SBATCH --output=./eulerlog/ft_array_job_slurm_%A_%a.out
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
# for take different input from different lines of input_file_list.txt
# echo $( awk "NR==$SLURM_ARRAY_TASK_ID" input_path_list.txt )

python tools/eval_baseline.py \
    --limit 5000 \
    --model vit \
    --skip_block $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/input_file_skip_block.txt )


# sbatch --array=1-12 jobArrayScript2.sh

# --dependency=afterany:341497

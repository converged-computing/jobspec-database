#!/usr/bin/env bash
#
#SBATCH --mail-user=zxu444@wisc.edu
#SBATCH --mail-type=ALL
#SBATCH -J vary_num  # give the job a name   
#***SBATCH --partition=batch_default ***
# 
# 1 node, 1 CPU per node (total 1 CPU), wall clock time of hours
#
#SBATCH -N 1                  ## Node count
#SBATCH --ntasks-per-node=1   ## Processors per node
#SBATCH --ntasks=1            ## Tasks
#SBATCH --gres=gpu:2          ## GPUs
#SBATCH --cpus-per-task=12    ## CPUs per task; number of threads of each task
#SBATCH -t 256:00:00          ## Walltime
#SBATCH --mem=40GB
#SBATCH -p lianglab
#SBATCH --error=./log/python_array_job_slurm_%A_%a.err
#SBATCH --output=./log/python_array_job_slurm_%A_%a.out
source ~/.bashrc

# Start GPU monitoring in the background
(
    while true; do
        nvidia-smi >> ./log/gpu/gpu_usage_${SLURM_JOB_ID}.log
        sleep 60  # Log every 60 seconds
    done
) &
monitor_pid=$!


echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

#*** for testing CUDA, run python code below
echo "======== testing CUDA available ========"
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
echo $( awk "NR==$SLURM_ARRAY_TASK_ID" input_path_list.txt )

#python finetune_maml.py \
#    --config $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/input_maml_ft_configs.txt )  \
#    --output_path $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/input_maml_ft_outpath.txt )

python test.py \
    --config $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/input_maml_test_configs.txt ) \
    --save_path $( awk "NR==$SLURM_ARRAY_TASK_ID" input_files_jobarray/input_maml_test_loadpath.txt ) \


# Kill the GPU monitoring process
kill $monitor_pid

### final running:
# sbatch --array=1-3 jobArrayScript.sh

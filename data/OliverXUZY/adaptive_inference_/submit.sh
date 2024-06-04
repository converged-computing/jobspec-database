#!/usr/bin/env bash
#
#SBATCH --output=./eulerlog/o_device_%j.out
#SBATCH --error=./eulerlog/o_device_%j.err
#SBATCH -J ada_inf  # give the job a name   
#***SBATCH --partition=batch_default ***
# 
# 1 node, 1 CPU per node (total 1 CPU), wall clock time of hours
#
#SBATCH -N 1                  ## Node count
#SBATCH --ntasks-per-node=1   ## Processors per node
#SBATCH --ntasks=1            ## Tasks
#SBATCH --gres=gpu:1          ## GPUs
#SBATCH --cpus-per-task=16     ## CPUs per task; number of threads of each task
#SBATCH -t 256:00:00          ## Walltime
#SBATCH --mem=80GB
#SBATCH -p lianglab
#SBATCH --exclude=euler[01-09],euler[11-12],euler[14],euler[24-27]
source ~/.bashrc

# Start GPU monitoring in the background
(
    while true; do
        nvidia-smi | tee -a ./log/gpu_usage_${SLURM_JOB_ID}.log
        sleep 600  # Log every 600 seconds
    done
) &
monitor_pid=$!


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




python train.py \
    -c '/srv/home/zxu444/vision/adaptive_inference/configs/resnet50_imagenet.yaml' \
    -n 'train_resnet50_imagenet' \
    -pf 1 \


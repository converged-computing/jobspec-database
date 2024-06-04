#!/bin/bash

# You must specify a valid email address!
#SBATCH --mail-user=<--YOUR-EMAIL-HERE-->

# Mail on NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-type=END

# Job name 
#SBATCH --job-name="ex_multi"
#SBATCH --array=1-3
#SBATCH --output=%x_%A_%a.out

# Runtime and memory
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
### 4 CPUs per GPU
#SBATCH --mem-per-cpu=4G

# Partition
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx3090:1

# Install dependencies #
singularity exec --nv docker://pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime pip install -U tensorboard
singularity exec --nv docker://pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime pip install -U scikit-learn

param_store=args2.txt
type=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')
backbone=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $2}')
fvs=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $3}')
image_modality=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $4}')
image_size=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $5}')
batch_size=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $6}')
suffix=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $7}')
# Run script #
# module load Workspace
singularity exec --nv docker://pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime python training.py -t $type -b $backbone -v $fvs -m $image_modality -i $image_size -s $batch_size -n $suffix
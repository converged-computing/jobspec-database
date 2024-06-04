#!/bin/bash

# CML SETUP


# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name=llm                                 # sets the job name if not set from environment
#SBATCH --array=0-9                                    # Submit 8 array jobs, throttling to 4 at a time
#SBATCH --output logs/opt_llama_%A_%a.log                            # indicates a file to redirect STDOUT to; %j is the jobid, _%A_%a is array task id
#SBATCH --error logs/opt_llama_%A_%a.log                             # indicates a file to redirect STDERR to; %j is the jobid,_%A_%a is array task id
#SBATCH --time=36:00:00                                         # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger                                    # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger                                         # set QOS, this will determine what resources can be requested
#SBATCH --gres=gpu:rtxa5000:4
#SBATCH --cpus-per-task=16
#SBATCH --partition=scavenger
#SBATCH --nice=0                                              #positive means lower priority
#SBATCH --exclude=legacygpu00,legacygpu01,legacygpu02,legacygpu03,legacygpu04,legacygpu05,legacygpu06,legacygpu07
#SBATCH --mem 100gb                                              # memory required by job; if unit is not specified MB will be assumed


source /cmlscratch/pchiang/miniconda3/etc/profile.d/conda.sh
conda activate hug

I=0
for data_fraction in 0.2 0.4 0.6 0.8 1.0; do
# training script for opt6.7b on 4 rtxa5000 for 5 epochs 52000*5/128=2031
# generate random port
model_size=opt6.7b
model_path=/fs/cml-projects/instruction_following/pretrained_models/${model_size}_sharded
model_config_path=/fs/cml-projects/instruction_following/pretrained_models/${model_size}_hf
name=${model_size}_self-instruct_dataf${data_fraction}_fsdp
ckpt_path=/fs/cml-projects/instruction_following/$name

command_list[$I]="python train.py --init_checkpoint_path $model_path \
--model_config_path $model_config_path \
--data_path data_instruct/alpaca.json --added_tokens 0 \
--act_checkpointing --lr 2e-5 --max_steps 2031 --accumulation_steps 4 --batch_size 8 \
--wandb --wb_project huggingface --wrapped_class_name OPTDecoderLayer \
--checkpoint_path $ckpt_path \
--wb_name $name --wb_id ${name}_v2 \
--data_fraction $data_fraction --hack"
I=$((I+1))

# training script for llama7B on 4 rtxa5000 for 5 epochs 52000*5/128=2031
model_size=7B
model_path=/fs/nexus-scratch/pchiang/llama/${model_size}_sharded
model_config_path=/fs/nexus-scratch/pchiang/llama/${model_size}_hf
name=llama${model_size}_self-instruct_dataf${data_fraction}_fsdp
ckpt_path=/fs/cml-projects/instruction_following/$name
command_list[$I]="python train.py --init_checkpoint_path $model_path \
--model_config_path $model_config_path \
--data_path data_instruct/alpaca.json --added_tokens 1 \
--act_checkpointing --lr 2e-5 --max_steps 2031 --accumulation_steps 8 --batch_size 4 \
--wandb --wb_project huggingface --wrapped_class_name LlamaDecoderLayer \
--checkpoint_path $ckpt_path \
--wb_name $name --wb_id ${name}_v2 \
--data_fraction $data_fraction --hack"

I=$((I+1))
done

echo ${command_list[$SLURM_ARRAY_TASK_ID]}
eval ${command_list[$SLURM_ARRAY_TASK_ID]}
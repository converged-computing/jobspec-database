#!/bin/bash
# Name of the job
#SBATCH -J 557flant5dst
# time: 48 hours
#SBATCH --time=999:0:0
# Number of GPU
#SBATCH --gres=gpu:rtx_6000_ada:2
# Number of cpus
#SBATCH --cpus-per-task=2
# Log output
#SBATCH -e ./log/slurm-err-%j.txt
#SBATCH -o ./log/slurm-out-%j.txt
#SBATCH --open-mode=append
#SBATCH --array=0-2
# Start your application
eval "$(conda shell.bash hook)"

N="$SLURM_ARRAY_TASK_ID"
conda activate adapter
TASKS=(english arabic french)
lang=${TASKS[N]}
saving_dir=output/$lang/flan-t5/small-357/5epochs/
python T5.py \
  --model_checkpoint 'google/flan-t5-small' \
  --model_name 'flan-t5' \
  --train_batch_size 4 \
  --GPU 2 \
  --seed 557\
  --slot_lang slottype \
  --n_epochs 5 \
  --saving_dir $saving_dir \
  --data_dir data/new_dst_$lang
#!/bin/bash
#
#SBATCH -p a40
#SBATCH --gres=gpu:2
#SBATCH -c 16                   # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     
#SBATCH -a 1
#SBATCH -t 4-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=100GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/sbatch_logs/%x_%j.out      # File to which STDOUT will be written
#SBATCH -e /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/sbatch_logs/%x_%j.err      # File to which STDERR will be written

# alphas=(0.001 0.1 0.5)
# DECOV_ALPHA=${alphas[(${SLURM_ARRAY_TASK_ID}-1)]}

srun --jobid $SLURM_JOBID bash -c 'python -m deep-learning-base.supervised_training \
--dataset imagenet \
--transform_dataset imagenet \
--save_every 0 \
--model resnet50 \
--batch_size 256 \
--wandb_name imagenet-training-scratch \
--max_epochs 50 \
--optimizer sgd \
--lr 0.01 \
--step_lr 500 \
--warmup_steps 1000 \
--gradient_clipping 1.0 \
--loss decov \
--decov_alpha ${SLURM_ARRAY_TASK_ID}'
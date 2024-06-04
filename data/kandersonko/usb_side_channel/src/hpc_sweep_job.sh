#!/usr/bin/env sh
#SBATCH --job-name=param-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p gpu-volatile
#SBATCH --gres=gpu:a6000:2
#SBATCH --time=08:00:00
#SBATCH --output=jobs/sweep_%A_%a.stdout
#SBATCH --error=jobs/sweep_%A_%a.stderr

cd $SLURM_SUBMIT_DIR

echo "JOB timestamp: $(date)"
echo "JOB ID: $SLURM_JOB_ID"
# echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

hostname
source ~/.bashrc

# module load cuda/11.8

nvidia-smi -L

# activate conda environment
conda activate usb2

python --version

which python

# fix issues of pytorch with cudnn library
# unset LD_LIBRARY_PATH

# Assign each task to a specific GPU
# export CUDA_VISIBLE_DEVICES=$SLURM_ARRAY_TASK_ID

# dataset_a
sweep_id=kxh8r98c

# # dataset_b
# sweep_id=etmmz8md

# # dataset_c1
# sweep_id=9x6kjm8p

# # dataset_c2
# sweep_id=z6bmeqd3

# # dataset_d1
# sweep_id=86di6s6s

# # dataset_d2
# sweep_id=ujblllhg

# srun --exclusive --gres=gpu:a100:2 -l wandb agent koffi-anderson/usb_side_channel/${sweep_id}
# srun --exclusive --gres=gpu:2 -l wandb agent koffi-anderson/usb_experiments/${sweep_id}
srun --exclusive --gres=gpu:2 -l wandb agent koffi-anderson/usb_experiments/${sweep_id}

#!/usr/bin/env sh
#SBATCH -N 1
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH -p gpu-8
#SBATCH --output=jobs/tsfresh_%A.stdout
#SBATCH --error=jobs/tsfresh_%A.stderr

cd $SLURM_SUBMIT_DIR

echo "JOB timestamp: $(date)"
echo "JOB ID: $SLURM_JOB_ID"


hostname

# source /usr/modules/init/bash
source ~/.bashrc

# activate conda environment
conda activate usb

python --version
which python

# module load cuda/11.8

# nvidia-smi -L

# fix issues of pytorch with cudnn library
# unset LD_LIBRARY_PATH

# Execute your script
# srun python tsfresh_feature_engineering.py
srun python tsfresh_feature_engineering.py \
    --target_label=category --dataset_subset=test \
    --workers=92 --memory='2GB' --chunk_size=10

#!/bin/bash
#
#SBATCH --job-name=translate-nematus
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-0-0
#SBATCH --time=72:00:00
#SBATCH --mem=20G
#SBATCH --output=log/slurm/translate_%a.out
#SBATCH --error=log/slurm/translate_%a.err

TASK_ID=$(printf '%0'$digits'd' $SLURM_ARRAY_TASK_ID)

echo $CUDA_VISIBLE_DEVICES
echo $TASK_ID

. /home/jwei/miniconda3/etc/profile.d/conda.sh
conda activate nematus
module load cuda/9.0.176
module load cudnn/7.0-cuda_9.0

./nematus/nematus/translate.py \
  --models data/translate/models/model.npz \
  -v -p 1 --n-best \
  --input data/translate/splits/test_splits/$TASK_ID \
  --output data/translate/output/output_splits/$TASK_ID \

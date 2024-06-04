#!/bin/bash

#SBATCH --job-name='dbg'
#SBATCH --partition=gpushort
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:v100
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --array=1
#--mail-type=FAIL,END
#--mail-user=t.n.van.der.werff@student.rug.nl

set -euo pipefail

module purge
module load Python/3.8.6-GCCcore-10.2.0
source ~/activate_py3.8.6

# pip install -e ..
# pip install -e ../htr

seed=$SLURM_ARRAY_TASK_ID

# learning_rate=3e-4
main_model_arch="WriterCodeAdaptiveModel"
base_model_arch="fphtr"
learning_rate=1e-5
weight_decay=0
shots=4
ways=4

# TRAINED_MODEL_PATH=/data/s4314719/full_page_HTR/lightning_logs/FPHTR_word_resnet18_lr=3e-4_bsz=32_seed=1/checkpoints/epoch=98-char_error_rate=0.1346-word_error_rate=0.1693.ckpt
TRAINED_MODEL_PATH=/data/s4314719/full_page_HTR/lightning_logs/FPHTR_word_resnet18_lr=3e-4_bsz=32_seed=3/checkpoints/epoch=64-char_error_rate=0.1500-word_error_rate=0.1738.ckpt


CACHE_DIR=/data/s4314719/thesis/writer_code/lightning_logs/cache_without_bad_segmentation

# cd $HOME/master-thesis/writer_code
srun python main.py \
--main_model_arch $main_model_arch \
--base_model_arch $base_model_arch \
--trained_model_path $TRAINED_MODEL_PATH \
--cache_dir $CACHE_DIR \
--learning_rate $learning_rate \
--weight_decay $weight_decay \
--seed $seed \
--shots $shots \
--ways $ways \
--data_dir /data/s4314719/IAM \
--max_epochs 1000 \
--num_workers 12 \
--track_grad_norm 2 \
--check_val_every_n_epoch 10000 \
--num_sanity_val_steps 0 \
# --overfit_batches 1 \
#--grad_clip 1 \
# --num_sanity_val_steps 1 \
#--use_cosine_lr_scheduler \
# --validate
# --save_all_checkpoints \

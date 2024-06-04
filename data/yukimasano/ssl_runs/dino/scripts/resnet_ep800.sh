#!/bin/bash
#SBATCH --nodes=10
#SBATCH --gpus=80
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --job-name=dino_resnet800
#SBATCH --constraint=volta32gb
#SBATCH --time=70:00:00
#SBATCH --mem=450G

module load anaconda3
source activate ssl_runs

DATADIR=$1
OUTDIR=$2
OUTDIR+='/dino_resnet800'
mkdir -p ${OUTDIR}

python -m torch.distributed.launch --nproc_per_node=8 main_dino.py \
         --world_size 80 --ngpus 8 --nodes 10 --optimizer lars --use_bn_in_head true \
         --arch resnet50 --weight_decay 1e-4 \
         --warmup_teacher_temp_epochs 50 \
         --clip_grad 0 --batch_size_per_gpu 51 --epochs 800 --freeze_last_layer 1 \
         --lr 0.2 --warmup_epochs 10 --min_lr 0.0048 \
         --weight_decay_end 0.000001 --global_crops_scale 0.14 1 \
         --out_dim 60000 \
         --use_fp16 false \
         --teacher_temp 0.07 \
         --norm_last_layer true \
         --local_crops_scale 0.05 0.14 --local_crops_number 6 \
         --data_path $DATADIR \
         --output_dir $OUTDIR
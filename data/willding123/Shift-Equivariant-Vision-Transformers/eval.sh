#!/bin/bash
# bash script for single node gpu training 
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus=a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
##SBATCH --gpus=a100_1g.5gb:1
#SBATCH --time=2:00:00
#SBATCH --job-name=eval1
#SBATCH --output=eval1.out
#SBATCH --error=eval1.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pding@umd.edu

module load cuda/11.6.2/
source ~/scratch.cmsc663/miniconda3/bin/activate
conda activate swin 
cd ~/scratch.cmsc663/Swin-Transformer 


# # twins base 1k
# python eval.py --model twins --model_card timm/twins_svt_base --data_path ~/scratch.cmsc663/val --batch_size 64 --write_csv --pretrained_path ~/scratch.cmsc663/twins_svts_b_scratch --ckpt_num 270
# python eval.py --model twins --model_card timm/twins_svt_base --data_path ~/scratch.cmsc663/val --batch_size 64 --write_csv --shift_attack --pretrained_path ~/scratch.cmsc663/twins_svts_b_scratch --ckpt_num 270
# python eval.py --model twins --model_card timm/twins_svt_base --data_path ~/scratch.cmsc663/val --batch_size 64 --write_csv --random_affine --pretrained_path ~/scratch.cmsc663/twins_svts_b_scratch --ckpt_num 270
# python eval.py --model twins --model_card timm/twins_svt_base --data_path ~/scratch.cmsc663/val --batch_size 64 --write_csv --random_perspective --pretrained_path ~/scratch.cmsc663/twins_svts_b_scratch --ckpt_num 270
# python eval.py --model twins --model_card timm/twins_svt_base --data_path ~/scratch.cmsc663/val --batch_size 64 --write_csv --crop --pretrained_path ~/scratch.cmsc663/twins_svts_b_scratch --ckpt_num 270
# python eval.py --model twins --model_card timm/twins_svt_base --data_path ~/scratch.cmsc663/val --batch_size 64 --write_csv --flip --pretrained_path ~/scratch.cmsc663/twins_svts_b_scratch --ckpt_num 270
# python eval.py --model twins --model_card timm/twins_svt_base --data_path ~/scratch.cmsc663/val --batch_size 64 --write_csv --random_erasing --pretrained_path ~/scratch.cmsc663/twins_svts_b_scratch --ckpt_num 270
# python eval.py --model twins --model_card timm/twins_svt_base --data_path ~/scratch.cmsc663/val --batch_size 64 --write_csv --all_attack --pretrained_path ~/scratch.cmsc663/twins_svts_b_scratch --ckpt_num 270


# # # polytwins base 1k
# python eval.py --model polytwins --model_card timm/twins_svt_base --data_path ~/scratch.cmsc663/val --batch_size 64 --write_csv --pretrained_path ~/scratch.cmsc663/ptwins_svts_b_scratch --ckpt_num 292
# python eval.py --model polytwins --model_card timm/twins_svt_base --data_path ~/scratch.cmsc663/val --batch_size 64 --write_csv --shift_attack --pretrained_path ~/scratch.cmsc663/ptwins_svts_b_scratch --ckpt_num 292
# python eval.py --model polytwins --model_card timm/twins_svt_base --data_path ~/scratch.cmsc663/val --batch_size 64 --write_csv --random_affine --pretrained_path ~/scratch.cmsc663/ptwins_svts_b_scratch --ckpt_num 292
# python eval.py --model polytwins --model_card timm/twins_svt_base --data_path ~/scratch.cmsc663/val --batch_size 64 --write_csv --random_perspective --pretrained_path ~/scratch.cmsc663/ptwins_svts_b_scratch --ckpt_num 292
# python eval.py --model polytwins --model_card timm/twins_svt_base --data_path ~/scratch.cmsc663/val --batch_size 64 --write_csv --crop --pretrained_path ~/scratch.cmsc663/ptwins_svts_b_scratch --ckpt_num 292
# python eval.py --model polytwins --model_card timm/twins_svt_base --data_path ~/scratch.cmsc663/val --batch_size 64 --write_csv --flip --pretrained_path ~/scratch.cmsc663/ptwins_svts_b_scratch --ckpt_num 292
# python eval.py --model polytwins --model_card timm/twins_svt_base --data_path ~/scratch.cmsc663/val --batch_size 64 --write_csv --random_erasing --pretrained_path ~/scratch.cmsc663/ptwins_svts_b_scratch --ckpt_num 292
# python eval.py --model polytwins --model_card timm/twins_svt_base --data_path ~/scratch.cmsc663/val --batch_size 64 --write_csv --all_attack --pretrained_path ~/scratch.cmsc663/ptwins_svts_b_scratch --ckpt_num 292

# # polyvit base from scratch
python eval.py --model polyvit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path ~/scratch.cmsc663/pvit_b_1kscratch --write_csv --ckpt_num 185
python eval.py --model polyvit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path ~/scratch.cmsc663/pvit_b_1kscratch --write_csv --shift_attack --ckpt_num 185
python eval.py --model polyvit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path ~/scratch.cmsc663/pvit_b_1kscratch --write_csv --random_affine --ckpt_num 185
python eval.py --model polyvit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path ~/scratch.cmsc663/pvit_b_1kscratch --write_csv --random_perspective --ckpt_num 185
python eval.py --model polyvit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path ~/scratch.cmsc663/pvit_b_1kscratch --write_csv --crop --ckpt_num 185
python eval.py --model polyvit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path ~/scratch.cmsc663/pvit_b_1kscratch --write_csv --flip --ckpt_num 185
python eval.py --model polyvit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path ~/scratch.cmsc663/pvit_b_1kscratch --write_csv --random_erasing --ckpt_num 185
python eval.py --model polyvit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path ~/scratch.cmsc663/pvit_b_1kscratch --write_csv --all_attack --ckpt_num 185
# # # imagenet-o
# # python eval.py --model polyvit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/imagenet-o --batch_size 64 --pretrained_path ~/scratch.cmsc663/pvit_b_1kscratch --write_csv --ckpt_num 185
# # python eval.py --model polyvit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/imagenet-o --batch_size 64 --pretrained_path ~/scratch.cmsc663/pvit_b_1kscratch --write_csv --shift_attack  --ckpt_num 185


# # vit base from scratch
# python eval.py --model vit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path ~/scratch.cmsc663/vit_b_1kscratch --write_csv --ckpt_num 86
# python eval.py --model vit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path ~/scratch.cmsc663/vit_b_1kscratch --write_csv --shift_attack --ckpt_num 86
# python eval.py --model vit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path ~/scratch.cmsc663/vit_b_1kscratch --write_csv --random_affine --ckpt_num 86
# python eval.py --model vit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path ~/scratch.cmsc663/vit_b_1kscratch --write_csv --random_perspective --ckpt_num 86
# python eval.py --model vit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path ~/scratch.cmsc663/vit_b_1kscratch --write_csv --crop --ckpt_num 86
# python eval.py --model vit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path ~/scratch.cmsc663/vit_b_1kscratch --write_csv --flip --ckpt_num 86
# python eval.py --model vit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path ~/scratch.cmsc663/vit_b_1kscratch --write_csv --random_erasing --ckpt_num 86
# python eval.py --model vit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path ~/scratch.cmsc663/vit_b_1kscratch --write_csv --all_attack --ckpt_num 86
# # python eval.py --model vit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/imagenet-o --batch_size 64 --pretrained_path ~/scratch.cmsc663/vit_b_1kscratch --write_csv --ckpt_num 86
# # python eval.py --model vit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/imagenet-o --batch_size 64 --pretrained_path ~/scratch.cmsc663/vit_b_1kscratch --write_csv --shift_attack --ckpt_num 86



# eval
# polyvit small from scratch IN1k
python eval.py --model polyvit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path ~/scratch.cmsc663/pvit_s_1kscratch --write_csv
python eval.py --model polyvit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path ~/scratch.cmsc663/pvit_s_1kscratch --write_csv --shift_attack
python eval.py --model polyvit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64  --pretrained_path ~/scratch.cmsc663/pvit_s_1kscratch --write_csv --random_affine
python eval.py --model polyvit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path ~/scratch.cmsc663/pvit_s_1kscratch --write_csv --random_perspective
python eval.py --model polyvit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path ~/scratch.cmsc663/pvit_s_1kscratch --write_csv --crop 
python eval.py --model polyvit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path ~/scratch.cmsc663/pvit_s_1kscratch --write_csv --flip 
python eval.py --model polyvit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path ~/scratch.cmsc663/pvit_s_1kscratch --write_csv --random_erasing
python eval.py --model polyvit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path ~/scratch.cmsc663/pvit_s_1kscratch --write_csv --all_attack
# # imagenet-o
# # python eval.py --model polyvit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/imagenet-o --batch_size 512 --pretrained_path ~/scratch.cmsc663/pvit_small_w --write_csv 
# # python eval.py --model polyvit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/imagenet-o --batch_size 512 --pretrained_path ~/scratch.cmsc663/pvit_small_w --write_csv --shift_attack


# vit small from scratch
python eval.py --model vit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path ~/scratch.cmsc663/vit_s_1kscratch --write_csv
python eval.py --model vit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path  ~/scratch.cmsc663/vit_s_1kscratch --write_csv --shift_attack
python eval.py --model vit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path  ~/scratch.cmsc663/vit_s_1kscratch --write_csv --random_affine
python eval.py --model vit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path  ~/scratch.cmsc663/vit_s_1kscratch --write_csv --random_perspective
python eval.py --model vit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path  ~/scratch.cmsc663/vit_s_1kscratch --write_csv --crop
python eval.py --model vit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path  ~/scratch.cmsc663/vit_s_1kscratch --write_csv --flip
python eval.py --model vit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path  ~/scratch.cmsc663/vit_s_1kscratch --write_csv --random_erasing
python eval.py --model vit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 64 --pretrained_path  ~/scratch.cmsc663/vit_s_1kscratch --write_csv --all_attack

# # imagenet-o
# python eval.py --model vit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/imagenet-o --batch_size 512 --pretrained_path  ~/scratch.cmsc663/vit_small_411_w --write_csv
# python eval.py --model vit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/imagenet-o --batch_size 512 --pretrained_path  ~/scratch.cmsc663/vit_small_411_w --write_csv --shift_attack

# #  polyvit small 1k
# python eval.py --model polyvit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --pretrained_path ~/scratch.cmsc663/pvit_small_428_w --write_csv --ckpt_num 233
# python eval.py --model polyvit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --pretrained_path ~/scratch.cmsc663/pvit_small_428_w --write_csv --shift_attack --ckpt_num 233
# python eval.py --model polyvit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --pretrained_path ~/scratch.cmsc663/pvit_small_428_w --write_csv --random_affine --ckpt_num 233
# python eval.py --model polyvit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --pretrained_path ~/scratch.cmsc663/pvit_small_428_w --write_csv --random_perspective --ckpt_num 233
# python eval.py --model polyvit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --pretrained_path ~/scratch.cmsc663/pvit_small_428_w --write_csv --crop --ckpt_num 233
# python eval.py --model polyvit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --pretrained_path ~/scratch.cmsc663/pvit_small_428_w --write_csv --flip --ckpt_num 233
# python eval.py --model polyvit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --pretrained_path ~/scratch.cmsc663/pvit_small_428_w --write_csv --random_erasing --ckpt_num 233
# python eval.py --model polyvit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --pretrained_path ~/scratch.cmsc663/pvit_small_428_w --write_csv --all_attack --ckpt_num 233
# python eval.py --model polyvit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/imagenet-o --batch_size 512 --pretrained_path ~/scratch.cmsc663/pvit_small_428_w --write_csv --ckpt_num 233
# python eval.py --model polyvit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/imagenet-o --batch_size 512 --pretrained_path ~/scratch.cmsc663/pvit_small_428_w --write_csv --shift_attack --ckpt_num 233

# # # polyvit base 1k
# python eval.py --model polyvit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --pretrained_path ~/scratch.cmsc663/pvit_base_429_w --write_csv --ckpt_num 36
# python eval.py --model polyvit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --pretrained_path ~/scratch.cmsc663/pvit_base_429_w --write_csv --shift_attack --ckpt_num 36
# python eval.py --model polyvit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --pretrained_path ~/scratch.cmsc663/pvit_base_429_w --write_csv --random_affine    --ckpt_num 36
# python eval.py --model polyvit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --pretrained_path ~/scratch.cmsc663/pvit_base_429_w --write_csv --random_perspective --ckpt_num 36
# python eval.py --model polyvit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --pretrained_path ~/scratch.cmsc663/pvit_base_429_w --write_csv --crop --ckpt_num 36
# python eval.py --model polyvit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --pretrained_path ~/scratch.cmsc663/pvit_base_429_w --write_csv --flip --ckpt_num 36
# python eval.py --model polyvit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --pretrained_path ~/scratch.cmsc663/pvit_base_429_w --write_csv --random_erasing   --ckpt_num 36
# python eval.py --model polyvit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --pretrained_path ~/scratch.cmsc663/pvit_base_429_w --write_csv --all_attack       --ckpt_num 36
# python eval.py --model polyvit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/imagenet-o --batch_size 512 --pretrained_path ~/scratch.cmsc663/pvit_base_429_w --write_csv --ckpt_num 36
# python eval.py --model polyvit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/imagenet-o --batch_size 512 --pretrained_path ~/scratch.cmsc663/pvit_base_429_w --write_csv --shift_attack  --ckpt_num 36

# # vit small 1k
# python eval.py --model vit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv
# python eval.py --model vit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --shift_attack
# python eval.py --model vit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --random_affine
# python eval.py --model vit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --random_perspective
# python eval.py --model vit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --crop
# python eval.py --model vit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --flip
# python eval.py --model vit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --random_erasing
# python eval.py --model vit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --all_attack
# python eval.py --model vit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/imagenet-o --batch_size 512 --write_csv
# python eval.py --model vit --model_card timm/vit_small_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/imagenet-o --batch_size 512 --write_csv --shift_attack

# # vit base 1k
# python eval.py --model vit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv
# python eval.py --model vit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --shift_attack
# python eval.py --model vit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --random_affine
# python eval.py --model vit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --random_perspective
# python eval.py --model vit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --crop
# python eval.py --model vit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --flip
# python eval.py --model vit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --random_erasing
# python eval.py --model vit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --all_attack
# python eval.py --model vit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/imagenet-o --batch_size 512 --write_csv
# python eval.py --model vit --model_card timm/vit_base_patch16_224.augreg_in1k --data_path ~/scratch.cmsc663/imagenet-o --batch_size 512 --write_csv --shift_attack

# # twins small 1k
# python eval.py --model twins --model_card timm/twins_svt_small --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv
# python eval.py --model twins --model_card timm/twins_svt_small --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --shift_attack
# python eval.py --model twins --model_card timm/twins_svt_small --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --random_affine
# python eval.py --model twins --model_card timm/twins_svt_small --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --random_perspective
# python eval.py --model twins --model_card timm/twins_svt_small --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --crop
# python eval.py --model twins --model_card timm/twins_svt_small --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --flip
# python eval.py --model twins --model_card timm/twins_svt_small --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --random_erasing
# python eval.py --model twins --model_card timm/twins_svt_small --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --all_attack


# # twins base 1k
# python eval.py --model twins --model_card timm/twins_svt_base --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv
# python eval.py --model twins --model_card timm/twins_svt_base --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --shift_attack
# python eval.py --model twins --model_card timm/twins_svt_base --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --random_affine
# python eval.py --model twins --model_card timm/twins_svt_base --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --random_perspective
# python eval.py --model twins --model_card timm/twins_svt_base --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --crop
# python eval.py --model twins --model_card timm/twins_svt_base --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --flip
# python eval.py --model twins --model_card timm/twins_svt_base --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --random_erasing
# python eval.py --model twins --model_card timm/twins_svt_base --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --all_attack

# vit base 21k
# python eval.py --model vit --model_card timm/vit_base_patch16_224 --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv
# python eval.py --model vit --model_card timm/vit_base_patch16_224 --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --shift_attack
# python eval.py --model vit --model_card timm/vit_base_patch16_224 --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --random_affine
# python eval.py --model vit --model_card timm/vit_base_patch16_224 --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --random_perspective
# python eval.py --model vit --model_card timm/vit_base_patch16_224 --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --crop
# python eval.py --model vit --model_card timm/vit_base_patch16_224 --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --flip
# python eval.py --model vit --model_card timm/vit_base_patch16_224 --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --random_erasing
# python eval.py --model vit --model_card timm/vit_base_patch16_224 --data_path ~/scratch.cmsc663/val --batch_size 512 --write_csv --all_attack

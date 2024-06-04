#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH --mem 1000 # 2GB solicitados.
#SBATCH -p mlow # or mlow Partition to submit to master low prioriy queue
#SBATCH --gres gpu:1 # Para pedir Pascales MAX 8
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written

# without refine =====================================================
python main.py \
--inference_dir demo/kitti \
--output_path output/gmflow-norefine-sintel_market_1 \
--resume pretrained/gmflow_kitti-285701a8.pth \

# with refine =====================================================
python main.py \
--inference_dir demo/kitti \
--output_path output/ \
--resume pretrained/gmflow_with_refine_kitti-8d3b9786.pth \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
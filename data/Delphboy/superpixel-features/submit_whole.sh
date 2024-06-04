#!/bin/bash

#SBATCH --chdir=/jmain02/home/J2AD007/txk47/hxs67-txk47/superpixel-features
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=small

module load cuda/10.2
module load python/anaconda3

# source .venv/bin/activate
conda activate superpixels

SIZE=$(echo "$SIZE" | bc)

model_id="ResNet"

# Generate whole image features
python3 main.py --image_dir /jmain02/home/J2AD007/txk47/hxs67-txk47/coco/${SET}2014/ \
		--save_dir /jmain02/home/J2AD007/txk47/hxs67-txk47/superpixel_features/${model_id}/whole_img \
		--model_id ${model_id} \
		--whole_img

python3 merge_and_clean.py --input_dir /jmain02/home/J2AD007/txk47/hxs67-txk47/superpixel_features/${model_id}/whole_img \
							--output_dir /jmain02/home/J2AD007/txk47/hxs67-txk47/superpixel_features/${model_id}/whole_img

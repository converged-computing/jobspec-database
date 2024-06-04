#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu
#SBATCH --job-name=M1
#SBATCH --output=test.out

module purge


singularity exec --nv \
            --overlay /scratch/yp2285/ws_nerf/overlay-50G-10M.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            /bin/bash -c "source /ext3/env.sh; python /scratch/yp2285/ws_nerf/detectron2/src/main.py --config-file /scratch/yp2285/ws_nerf/detectron2/projects/MViTv2/configs/cascade_mask_rcnn_mvitv2_b_in21k_3x.py --input_data images --output_path results --eval-only --opts train.init_checkpoint=/scratch/yp2285/ws_nerf/detectron2/checkpoints/model_final_be5168.pkl"
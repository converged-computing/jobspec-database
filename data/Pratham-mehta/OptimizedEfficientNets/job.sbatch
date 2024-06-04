#!/bin/bash
#SBATCH --job-name=HPML_Project_efficientnet_b0_dataparallel_finetune_GPU_1_with_profiler_new_image_size_food
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x_%j.out

module purge
cd /scratch/pm3483/Project1/

singularity exec --nv \
      --overlay /scratch/pm3483/pytorch-example/overlay-25GB-500K.ext3:ro \
     /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; python train.py --seed 42 --arch efficientnet_b0 --epochs 4 --dataset food --profile False"

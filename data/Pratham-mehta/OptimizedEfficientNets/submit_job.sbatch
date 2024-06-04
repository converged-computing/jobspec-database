#!/bin/bash
#SBATCH --job-name=HPML_Project_efficientnet_b1_dataparallel
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x_%j.out

module purge

cd /scratch/sd5023/HPML/Course_Project/

singularity exec --nv \
      --overlay /scratch/sd5023/video_vpr/overlay-15GB-500K.ext3:ro \
     /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; python train.py --seed 42 --arch efficientnet_b1"
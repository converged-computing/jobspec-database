#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=PIPE
#SBATCH --output=%x.out

module purge

singularity exec --nv \
            --overlay /scratch/dy2242/my_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh;
            python main_PIPE.py
            "
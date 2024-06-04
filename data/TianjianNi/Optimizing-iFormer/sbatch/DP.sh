#!/bin/bash
#SBATCH --partition=rtx8000
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=DP
#SBATCH --output=DP

module purge
singularity exec --nv \
            --overlay /scratch/tn2151/pytorch-example/overlay-10GB-400K.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh;
        python main_DP.py --batch_size 512 --num_workers 16;"

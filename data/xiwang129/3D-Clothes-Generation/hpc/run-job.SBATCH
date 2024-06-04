#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output="sp-gan-output.out"
#SBATCH --job-name=python-training

module purge

singularity \
    exec --nv \
    --overlay /scratch/xw914/cv-final-flow/overlay-7.5GB-300K.ext3:ro \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "
source /ext3/env.sh
python modelTrain.py > log.txt
"

#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --time=24:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu

#SBATCH --job-name=train
#SBATCH --output=train.out

##SBATCH --mail-type=END
##SBATCH --mail-user=tw2672@nyu.edu

module purge

singularity exec --nv --overlay /scratch/tw2672/pytorch/torch2cuda8.ext3:ro  /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif  /bin/bash -c 'source /ext3/env.sh;python main.py'
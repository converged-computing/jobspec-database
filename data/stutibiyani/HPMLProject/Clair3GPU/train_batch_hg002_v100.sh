#!/bin/bash
#SBATCH --job-name=clair3_train_hg002_v100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --output=%x.out
#SBATCH --gres=gpu:4
#SBATCH --partition=v100
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --mail-user=sb7580@nyu.edu
#SBATCH --mail-type=ALL

singularity exec --nv --overlay /scratch/sc8781/Project/environment/overlay-50G-10M.ext3:rw \
 /scratch/work/public/singularity/cuda9.2.148-cudnn7.6.5.32-devel-ubuntu18.04.6.sif /bin/bash \
 -c "source /ext3/env.sh && /scratch/sb7580/HPMLProject/Clair3Proj/pileup_training_hg002.sh && /bin/bash -norc;"

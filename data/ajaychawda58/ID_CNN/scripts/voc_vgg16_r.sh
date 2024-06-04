#!/bin/bash
#SBATCH --mem-per-cpu=16G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:V100:1

module load nvidia/11.1
module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh
conda activate idenv

python ../train.py --dataset pascal_voc --backbone vgg16 --model retinanet --epoch 60

conda deactivate

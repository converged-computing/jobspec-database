#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=stephen.krewson@yale.edu
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH --mem-per-cpu=8g
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:2
#SBATCH --gres-flags=enforce-binding

# was using -n instead of -c! (use squeue to check computer nodes)
# to check versions: module spider Matlab

module purge
module restore cuda
source deactivate
source activate maskRCNN

# start with weights=coco; to resume training use weights=last
python balloon.py train --dataset=../../datasets/balloon --weights=last


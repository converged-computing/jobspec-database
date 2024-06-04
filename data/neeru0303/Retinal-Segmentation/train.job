#!/bin/sh

#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --mail-type=ALL
#SBATCH --mail-user=addankn@sunyit.edu
#SBATCH --gres=gpu:p100:2
#SBATCH -t 36:30:00

#echo commands to stdout
set -x

export CUDA_VISIBLE_DEVICES=0,1

module load tensorflow/1.5_gpu
module load keras/2.0.4

export TENSORFLOW_ENV=$TF_ENV

source $KERAS_ENV/bin/activate

#move to working directory
cd $HOME

cd r2/Retinal-Segmentation

# python python/create_model_upsample.py --cache
python python/main_model.py --classification 4 --dataset big --cache --activation relu


echo "ALL DONE!"

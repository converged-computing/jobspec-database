#!/bin/bash

# Slurm Resource Parameters (Example)
#SBATCH -t 5-00:00              # Runtime in D-HH:MM
#SBATCH -p gpu                  # Partition to submit to
#SBATCH --gres=gpu:1            # Number of gpus
#SBATCH -w clamps
#SBATCH -o attention_gan_%j.out      # File to which STDOUT will be written
#SBATCH -e attention_gan_%j.err      # File to which STDERR will be written

srun nvidia-docker run --rm -e CUDA_VISIBLE_DEVICES=`echo $CUDA_VISIBLE_DEVICES` -v /home/$USER:/home/$USER madratman/theano_keras_cuda8_cudnn5 python /home/ratneshm/projects/attention_gan/model.py

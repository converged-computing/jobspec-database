#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3 -c3
#SBATCH --time=05:00:00
#SBATCH --mem=240GB
#SBATCH --job-name=happyDB12
#SBATCH --mail-type=END
#SBATCH --mail-user=jx603@nyu.edu
#SBATCH --output=slurm_%j.out

# This are the hyperparameters you can change to fit your data
module purge
module load numpy/python3.6/intel/1.14.0 tensorflow/python3.6/1.5.0

cd /scratch/jx603/training-lstm-master2

python train.py --data_dir=./data \
--rnn_size 1024 \
--num_layers 2 \
--seq_length 128 \
--batch_size 128 \
--num_epochs 50 \
--save_checkpoints ./checkpoints \
--save_model ./models
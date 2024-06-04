#!/bin/bash
#PBS -N DeepUbuntu
#PBS -A jim-594-aa
#PBS -l walltime=12:00:00
#PBS -l nodes=1:gpus=2

module load apps/python/2.7.5

cd "${PBS_O_WORKDIR}"
#cd Ubuntu/ubottu/src
source lasagne/bin/activate
THEANO_FLAGS='floatX=float32,device=gpu' python pos_tag.py --n_recurrent_layers 2 --seq_len 96 --batch_size 128 --model_name context_2_LSTM_2 --n_epochs 5 --load_file LSTM_2_1.000000_0_adam_0.968693.pkl 

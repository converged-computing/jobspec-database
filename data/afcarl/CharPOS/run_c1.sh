#!/bin/bash
#PBS -N DeepUbuntu
#PBS -A jim-594-aa
#PBS -l walltime=12:00:00
#PBS -l nodes=1:gpus=2

module load apps/python/2.7.5

cd "${PBS_O_WORKDIR}"
#cd Ubuntu/ubottu/src
source lasagne/bin/activate
THEANO_FLAGS='floatX=float32,device=gpu' python pos_tag.py --n_recurrent_layers 1 --suffix _ctx1_b256_sq48_shTrue --seq_len 48 --batch_size 256 --model_name context_1_LSTM_1 --n_epochs=10

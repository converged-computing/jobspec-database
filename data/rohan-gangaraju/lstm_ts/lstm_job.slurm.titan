#!/bin/bash
#
#SBATCH --partition=titanx-long    # Partition to submit to <m40-short|m40-long|teslax-short|teslax-long>
#SBATCH --job-name=lstm
#SBATCH -o lstm_res_%j.txt            # output file
#SBATCH -e lstm_res_%j.err            # File to which STDERR will be written
#SBATCH --ntasks=1
#SBATCH --time=02-01:00:00          # D-HH:MM:SS
#SBATCH --gres=gpu:1             # Very Imp. Without this, CUDA does not recognise GPUs

source /home/rgangaraju/.bashrc

source activate tf

#lspci -vnn|grep NVIDIA

hostname

#python -u mnist-deep.py
python -u /home/rgangaraju/lstm/generic_lstm.py
#python -u /home/rgangaraju/lstm/create_graph.py

sleep 1
exit

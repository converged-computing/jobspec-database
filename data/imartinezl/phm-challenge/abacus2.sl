#!/bin/bash
#SBATCH -J main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=14-00:00:00
#SBATCH --partition=gpu
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --nodelist=abacus002
#SBATCH --cpus-per-task=8

# salloc --ntasks=1 --partition=interactive --time=1-00:00:00  --mem=8GB --gpus=1 --nodelist=abacus002 --cpus-per-task=8

module load Miniconda3/4.9.2
# if ! (conda env list | grep ".venv2") ; then 
# 	conda create --name .venv2 python=3.7 -y
# fi
module load CUDA/11.1.1-GCC-10.2.0
source activate .venv2
export CXX=g++
# pip install -r requirements.txt

# pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
# pip3 install difw

srun python run.py

# srun python main.py --folder results --batch-size 64 --rnn-num-layers 8 --rnn-hidden-size 64 --fcn-hidden-layers 4 --fcn-hidden-size 64  --embedding-size 10 --split 0.7 --epochs 1 --lr 0.0001

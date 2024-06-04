#!/bin/bash
#SBATCH --time=70:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=dl2_e_greedy_pong
#SBATCH --mem=72000
module load Python/3.6.4-foss-2018a
module load CUDA/9.1.85
module load Boost/1.66.0-foss-2018a-Python-3.6.4
module load TensorFlow/1.12.0-fosscuda-2018a-Python-3.6.4
#pip install --user pygame
#pip install --user keras
#pip install --user matplotlib
python ./pong.py
mv *.out slurm/

#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=30G
#SBATCH --job-name=DCGAN

source /data/$USER/.envs/pyenv37/bin/activate

module load Python/3.7.4-GCCcore-8.3.0 
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4

echo Starting Python program

python /data/s3972445/.envs/pyenv37/dreaming-rl/DCGAN.py
 








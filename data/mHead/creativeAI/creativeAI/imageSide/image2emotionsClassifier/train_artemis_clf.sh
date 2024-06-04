#!/bin/bash
#SBATCH --job-name=creativeAI-image2emotionClassifier
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --partition=cuda
#SBATCH --mem=50GB
#SBATCH --time=24:00:00
#SBATCH --output=creativeAI_st_%j_out.txt
#SBATCH --error=creativeAI_st_%j_err.txt

#clean the module environment may inherited from the calling session
ml purge

#load modules
ml nvidia/cudasdk/10.1
ml intel/python/3/2019.4.088

cd /home/mtesta/creativeAI/imageSide

python3 main.py

srun --partition=cuda --nodes=1 --tasks-per-node=1 --gres=gpu:1 --time=06:00:00 --pty /bin/bash
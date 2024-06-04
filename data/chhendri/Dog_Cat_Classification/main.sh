#!/bin/bash
# Submission script for Dragon2
#SBATCH --job-name=project_ia
#SBATCH --time=24:00:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --gres="gpu:1"
#SBATCH --mem-per-cpu=2625 # megabytes
#SBATCH --partition=gpu
#
#SBATCH --mail-user=Chris.Adam@ulb.be
#SBATCH --mail-type=ALL
#
#SBATCH --comment=techniques_of_ai_exam

module load TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4
python3 main.py
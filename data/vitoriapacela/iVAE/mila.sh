#!/bin/bash

# sbatch will read any lines that start with "#SBATCH" and will
# add what follows as if they were command line parameters.

#SBATCH --job-name=ivae
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=100Gb

# Load the necessary modules
module load miniconda/3 pytorch/1.8.1

# Activate a virtual environment, if needed:
conda activate research

# Run the script
#python fastica.py --obs_data_path data_5_40_1000_1.csv --mix_data_path mix_5_40_1000_1.csv --s 0 --config binary-6-2-lbfgs-100-seg.yaml --ckpt_folder='run/checkpoints/'

python main.py --config continuous-2-2-lbfgs.yaml --n-sims 1 --m 2.0 --s 0 --ckpt_folder='run/checkpoints/'

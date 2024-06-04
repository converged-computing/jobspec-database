#!/bin/bash
#SBATCH --job-name=HN_17
#SBATCH --qos=normal
#SBATCH -c 6
#SBATCH --mem=20G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=13:59:00
#SBATCH --output=HN_17/slurm-%j.out
#SBATCH --error=HN_17/slurm-%j.err

# Environment Setup
module purge
module load python/3.12.0
pip3 install --upgrade pip
pip3 install -U -q pandas numpy tensorflow cuda-python torch torchvision seaborn plotly matplotlib ipywidgets tqdm

# Run Experiments
python3 main.py --data_index 17

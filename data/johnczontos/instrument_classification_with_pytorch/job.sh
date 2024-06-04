#!/bin/bash
#SBATCH -w cn-m-1
#SBATCH -A soundbendor
#SBATCH -p soundbendor
#SBATCH --job-name=MIC
#SBATCH -t 1-00:00:00
#SBATCH -c 6
#SBATCH --mem=128G
#SBATCH --gres=gpu:6
#SBATCH --export=ALL

#SBATCH -o logs/run_model.out
#SBATCH -e logs/run_model.err

# load any software environment module required for app (e.g. matlab, gcc, cuda)
module load python/3.10 cuda/11.7 sox

# load env
source env/bin/activate

# run model
python run_model.py data/rwc_all/clean/split
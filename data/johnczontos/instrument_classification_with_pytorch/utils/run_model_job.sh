#!/bin/bash
#SBATCH -w cn-m-2
#SBATCH -p cascades
#SBATCH -A cascades
#SBATCH --job-name=instrument
#SBATCH -t 3-00:00:00
#SBATCH -c 8
#SBATCH --mem=124G
#SBATCH --gres=gpu:2
#SBATCH --export=ALL

#SBATCH -o logs/run_model.out				  # name of output file for this submission script
#SBATCH -e logs/run_model.err				  # name of error file for this submission script

# load any software environment module required for app (e.g. matlab, gcc, cuda)
module load python/3.10 cuda/11.7 sox

# move to split dataset dir
cd /nfs/guille/eecs_research/soundbendor/zontosj/instrument_classification_with_pytorch

# load env
source env/bin/activate

# run model
python run_model.py data/rwc_all/clean/split
#!/bin/bash
#SBATCH --time=0-24:00                  # How long you want to run for
#SBATCH --account=def-amw8          # The account you want to bill, if you have access to Martha's it's rrg-whitem or def-whitem. I don't know what Adam's is.
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name=relgan-tf-no-gpu
#SBATCH --output=/home/jcoblin/projects/def-amw8/jcoblin/RelGAN/logs/%x-output-no-gpu-%j.log
#SBATCH --error=/home/jcoblin/projects/def-amw8/jcoblin/RelGAN/logs/%x-error-no-gpu-%j.log

source .venv/relgan/bin/activate
module load python/3.7
# pip install torch transformers numpy tensorboard tqdm

cd oracle/experiments
echo "Current working directory: `pwd`"
echo "Running main.py"
python oracle_relgan.py $SLURM_ARRAY_TASK_ID -1
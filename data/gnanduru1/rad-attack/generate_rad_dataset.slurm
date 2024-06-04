#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=cs6501_sp24
# #SBATCH --partition=bii-gpu
# #SBATCH --account=bii_dsc_community
# #SBATCH --reservation=bi_fox_dgx

#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=rad-generate-dataset
#SBATCH --output=%u-%j-data.out
#SBATCH --error=%u-%j-data.err
#SBATCH --mem=256G
#SBATCH --array=0-3

date
nvidia-smi

source env.sh
python generate_rad_dataset.py --id $SLURM_ARRAY_TASK_ID --alpha=100 --n 3000

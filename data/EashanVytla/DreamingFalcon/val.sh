#!/bin/bash
#SBATCH --job-name=DreamerTraining_Prelim
#SBATCH --nodes=1 --ntasks-per-node=28 --gpus-per-node=1 --gpu_cmode=shared
#SBATCH --time=1:00:00
#SBATCH --account=PAS2152
#SBATCH --mail-type=ALL

cd $SLURM_SUBMIT_DIR

module load miniconda3/23.3.1-py310  cuda/12.3.0

source activate pytorch

python3 valid_dynamics.py
python3 test_dynamics.py
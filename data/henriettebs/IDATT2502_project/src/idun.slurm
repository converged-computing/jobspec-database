#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --account=ie-idi
#SBATCH --time 04:00:00
#SBATCH --nodes=1
#SBATCH -c 28
#SBATCH --mem=12000
#SBATCH --job-name="Stock prediction training"
#SBATCH --output=prediction-srun.out
#SBATCH --mail-user=hannagn@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}

cd ${WORKDIR}
module load Python/3.10.4-GCCcore-11.3.0
pip install numpy
pip install torch
pip install yahoo_fin
pip install tensorflow
pip install scikit_learn
pip install keras
pip install matplotlib
pip install pandas
pip install statsmodels
pip install yfinance
python3 src/main.py
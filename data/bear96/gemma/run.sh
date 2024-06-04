#!/bin/bash


#SBATCH -J run-function
#SBATCH --account=thermaltext
#SBATCH --partition=dgx_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --time=0-00:60:00
#SBATCH --gres=gpu:1

module load Python/3.11.3-GCCcore-12.3.0
python3 --version
nvidia-smi
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
python main.py

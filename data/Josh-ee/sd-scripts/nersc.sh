#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J sd-script
#SBATCH --mail-user=gaoyang29@berkeley.edu
#SBATCH --mail-type=all
#SBATCH -t 23:59:00
#SBATCH -A m4633

module load pytorch
source /pscratch/sd/y/yanggao/sd-scripts/venv/bin/activate
module load pytorch

python run.py
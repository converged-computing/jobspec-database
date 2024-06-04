#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --account=project_2002029
#SBATCH --mem=14G
#SBATCH -e train2.err -o train2.out
#SBATCH --gres=gpu:v100:1
#SBATCH -J cf42

LR=$1

module purge
module load pytorch/1.3.0
PROJDIR=$HOME/proj_deepsequence/scratch/ginter/cafa
source $PROJDIR/venv-torch/bin/activate
export PYTHONPATH=../venv-torch/lib64/python3.7/site-packages:$PYTHONPATH
python3 train.py --train CAFA4-ctrl/train.torchbin --dev CAFA4-ctrl/devel.torchbin --max-labels 5000 --class-stats-file CAFA4-ctrl/class-stats.json --store-cpoint fix-norank-checkpoint-CAFA4-ctrl.$LR --lrate $LR --report-every 1000

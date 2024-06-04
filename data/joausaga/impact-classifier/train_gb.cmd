#!/bin/bash
#SBATCH--job-name="social-impact-research-model-training"
#SBATCH--workdir=.
#SBATCH--output=train_gb_%j.out
#SBATCH--error=train_gb_%j.err
#SBATCH--ntasks=1
#SBATCH--time=24:00:00
#SBATCH--cpus-per-task=24

echo $(date '+%Y-%m-%d %H:%M:%S')
module purge
module load intel
module load mkl
module load python/3.7.4
source env/bin/activate
python train.py train-model GB './data/train/data_tc_2000_1_ 3_clean.pkl' --algorithm_name 'gradient_boosting' --metric 'f1'
echo $(date '+%Y-%m-%d %H:%M:%S')
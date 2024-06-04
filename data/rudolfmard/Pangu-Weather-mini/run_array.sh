#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --mem=100G
#SBATCH --job-name=RMSE
#SBATCH --output=RMSE_%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --constraint='volta'
#SBATCH --array=0-3

case $SLURM_ARRAY_TASK_ID in
    0) PROP=1.0; TRIGGER=0.0; LABEL=1.0; CHECKPOINT=results/poison-005_trigger-00_label-05_ceedfe; RESULTS=results/poison-005_trigger-00_label-05_ceedfe;;
    1) PROP=1.0; TRIGGER=1.0; LABEL=1.0; CHECKPOINT=results/poison-005_trigger-00_label-05_ceedfe; RESULTS=results/poison-005_trigger-00_label-05_ceedfe;;
    2) PROP=1.0; TRIGGER=0.2; LABEL=1.0; CHECKPOINT=results/poison-005_trigger-02_label-05_aef8a0; RESULTS=results/poison-005_trigger-02_label-05_aef8a0;;
    3) PROP=1.0; TRIGGER=1.0; LABEL=1.0; CHECKPOINT=results/poison-005_trigger-02_label-05_aef8a0; RESULTS=results/poison-005_trigger-02_label-05_aef8a0;;
esac

module load anaconda
source activate weather_model_env

python main.py -P $PROP -G $TRIGGER -B $LABEL -T -c $CHECKPOINT -r $RESULTS
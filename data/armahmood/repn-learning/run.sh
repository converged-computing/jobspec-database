#!/bin/bash
#SBATCH --array=0-575
#SBATCH --job-name=repn_learning   
#SBATCH --time=1:10:00
#SBATCH --mem-per-cpu=4000M
#SBATCH --account=def-ashique
#SBATCH --output=repn_learning%A%a.out
#SBATCH --error=repn_learning%A%a.err

python learner_xrel.py --search --save_losses --cfg ./cfg_temp/$SLURM_ARRAY_TASK_ID.json

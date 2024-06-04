#!/bin/bash -l
#SBATCH -o ./logs/tjob.out.%A_%a
#SBATCH -e ./logs/tjob.err.%A_%a
#SBATCH --job-name=RL3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=8

cd ~/RL3NeurIPS/

module purge
conda activate pytorch-gpu

# python3 fit.py  --recompute --full --prior svdo --subject ${SLURM_ARRAY_TASK_ID} --changepoint
# fit policy parameters for all subtasks
python3 fit.py --full --prior svdo --subject ${SLURM_ARRAY_TASK_ID} --changepoint
# fit policy parameters for last subtask
#python3 fit.py --prior svdo --subject ${SLURM_ARRAY_TASK_ID} --changepoint
# sbatch --array=0-109 experiments/fit_changepoint_entropy.sh
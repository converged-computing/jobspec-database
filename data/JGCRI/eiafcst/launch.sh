#!/bin/sh
#SBATCH -A eiafcst
#SBATCH -p shared
#SBATCH --gres=gpu:1
#SBATCH -n 8
#SBATCH -J eiafcst
#SBATCH -t 12:00:00

# Example command to run this script:
# 		sbatch --array=0-31 launch.sh 16 3 hpar_results

module purge
module load cuda/9.2.148
module load python/anaconda3.2019.3
source /share/apps/python/anaconda3.2019.3/etc/profile.d/conda.sh

nvidia-smi

# When running with --array, the environment variable $SLURM_ARRAY_TASK_ID is
# set uniquely for each run
tid=$SLURM_ARRAY_TASK_ID

echo "python eiafcst/models/hpar_opt.py $1 $2 $3${tid}.csv"
python eiafcst/models/hpar_opt.py $1 $2 $3${tid}.csv

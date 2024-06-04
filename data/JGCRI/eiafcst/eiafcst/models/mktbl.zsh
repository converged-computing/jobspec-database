#!/bin/zsh
#SBATCH -A eiafcst
#SBATCH -p shared
#SBATCH --gres=gpu:1
#SBATCH -J eiafcst
#SBATCH -t 12:00:00

# Example command to run this script:
# 		sbatch --array=0-31 launch.sh 16 3 hpar_results

module purge
module load cuda/9.2.148
module load python/anaconda3

nvidia-smi

# When running with --array, the environment variable $SLURM_ARRAY_TASK_ID is
# set uniquely for each run
tid=$SLURM_ARRAY_TASK_ID

(( end = $tid * 3 ))

files=`head -$end filelist.txt | tail -3`

echo "Running files:"
echo $files

echo $files | xargs python gather_model_stats.py > diagnostic/gdp/rslts-summary$tid.csv


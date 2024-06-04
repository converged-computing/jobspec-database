#!/bin/bash
#
#SBATCH --job-name=parameter_search_job
#SBATCH --output=parameter_search_job_output.%A_%a.txt
#
#SBATCH --ntasks=1
#SBATCH --array=0-11

docker1 run biohpc_wl428/xgb-shap-plt python /workdir/parameter_search.py $SLURM_ARRAY_TASK_ID
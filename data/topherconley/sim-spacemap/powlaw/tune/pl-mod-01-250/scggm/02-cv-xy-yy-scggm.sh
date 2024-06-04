#!/bin/bash -l
#SBATCH --job-name=02-sc
#SBATCH --mem-per-cpu=1120
# #SBATCH --array=1-24,26-29,31-100
#SBATCH --array=56-100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=17
#SBATCH -o /home/cconley/scratch-data/sim-spacemap/powlaw/pl-mod-01/results/n250/scggm/02-cv-xy-yy-scggm/logs/job_${SLURM_ARRAY_TASK_ID}.out
#SBATCH -e /home/cconley/scratch-data/sim-spacemap/powlaw/pl-mod-01/results/n250/scggm/02-cv-xy-yy-scggm/logs/job_${SLURM_ARRAY_TASK_ID}.err
#SBATCH -D /home/cconley/repos/sim-spacemap/powlaw/tune/pl-mod-01-250/scggm/
set -e
set -v
module add matlab
/usr/bin/R --no-save --no-restore --no-site-file --no-init-file  --args ${SLURM_ARRAY_TASK_ID} 17 < 02-cv-xy-yy-scggm.R 

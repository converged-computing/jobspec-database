#!/bin/bash -l

#SBATCH --job-name=scggm
#SBATCH --mem-per-cpu=1520
#SBATCH --array=1-100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=31
#SBATCH -o /home/cconley/scratch-data/sim-spacemap/powlaw/pl-mod-01/results/n250/scggm/01-cv-xy-yy-scggm/logs/%j.out
#SBATCH -e /home/cconley/scratch-data/sim-spacemap/powlaw/pl-mod-01/results/n250/scggm/01-cv-xy-yy-scggm/logs/%j.err
#SBATCH -D /home/cconley/repos/sim-spacemap/powlaw/tune/pl-mod-01-250/scggm 
set -e
set -v
module add matlab
/usr/bin/R --no-save --no-restore --no-site-file --no-init-file  --args ${SLURM_ARRAY_TASK_ID} 31 < 01-cv-xy-yy-scggm.R 

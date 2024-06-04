#!/bin/bash
#
# usage: sbatch run_pull_stats filter
#
# where filter = {1..6}
# 1 = descending (night), land & ocean
# 2 = desc, ocean
# 3 = desc, land
# 4 = ascending (day, land & ocean
# 5 = asc, ocean
# 6 = asc, land

# sbatch options
#SBATCH --job-name=RUN_IASI_PULL_STATS_CLR
# partition = dev/batch
#SBATCH --partition=high_mem
# qos = short/normal/medium/long/long_contrib
#SBATCH --qos=medium+
#SBATCH --account=pi_strow
#SBATCH -N1
#SBATCH --mem=28000
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --array=0-12

#SBATCH --mail-user=sbuczko1@umbc.edu
##SBATCH --mail-type=BEGIN
##SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=TIME_LIMIT_50

#SBATCH -o /home/sbuczko1/LOGS/sbatch/pull_stats_iasi_clr-%A_%a.out
#SBATCH -e /home/sbuczko1/LOGS/sbatch/pull_stats_iasi_clr-%A_%a.err

# matlab options
MATLAB=matlab
MATOPT=' -nojvm -nodisplay -nosplash'

JOBSTEP=0

echo "Executing srun of run_pull_stats"
$MATLAB $MATOPT -r "addpath('~/git/pull_stats_DEV/iasi');\
                    addpath('~/git/rtp_prod2_DEV/util');\
                    addpath('~/git/pull_stats_DEV/util');\
                    addpath('~/git/pull_stats_DEV/iasi/util');\
                    cfg=ini2struct('$2');\
                    run_pull_stats($1, cfg);\
                    exit"
    
echo "Finished with srun of run_pull_stats"




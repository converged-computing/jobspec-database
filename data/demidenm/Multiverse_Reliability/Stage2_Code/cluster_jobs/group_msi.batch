#!/bin/bash -l
#SBATCH -J group_abcd
#SBATCH --array=0-89 #59 or 89 # jobs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH -t 00:20:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdemiden@umn.edu
#SBATCH -p agsmall,msismall #,amdsmall #update for UMN/sherlock
#SBATCH -o log_abcd/%x_%A_%a.out #update for abcd/ahrb/mls
#SBATCH -e log_abcd/%x_%A_%a.err #update for abcd/ahrb/mls
#SBATCH -A faird #feczk001 # update for UMN/sherlock

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate fmri_env

ID=${SLURM_ARRAY_TASK_ID}
bash ./batch_jobs/group${ID}

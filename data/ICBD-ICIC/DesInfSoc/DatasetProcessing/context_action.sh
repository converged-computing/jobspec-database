#!/bin/bash
#SBATCH --job-name=itrust-context
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-21
#SBATCH --time=01-00:00:00
#SBATCH --output=outputs/context-%A-%a.out
#SBATCH --error=errors/context-%A-%a.err

source .context_env/bin/activate

srun python context_action.py ${SLURM_ARRAY_TASK_ID}
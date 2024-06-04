#!/bin/bash
#SBATCH --account=def-ichiro
#SBATCH --time=5-00:00
#SBATCH --array=97
#SBATCH --output=run_output/2_output_%a.out
#SBATCH --cpus-per-task=4  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=12500M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.

module load python/3.8
module load scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install google==3.0.0
pip install openpyxl

echo "Starting Task $SLURM_ARRAY_TASK_ID"
python -u data_lyrics.py --id $SLURM_ARRAY_TASK_ID

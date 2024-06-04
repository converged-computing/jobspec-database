#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=1-200 
#SBATCH -t 72:00:00
#SBATCH -o logs/%A_%a.out
#SBATCH -e logs/%A_%a.err

echo `hostname`

wdir=/projects/sequence_analysis/vol3/bizon/scigraph_cord
cd $wdir

conda activate translator

python co_occur.py output $SLURM_ARRAY_TASK_ID 200

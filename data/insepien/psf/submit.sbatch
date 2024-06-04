#!/bin/bash -l
#SBATCH --partition=kipac
#SBATCH -t 2:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mydo@stanford.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=6G
#SBATCH --output=submit.h%a.out
#SBATCH --array=91-130

source ~/setup.sh
conda activate wfsim

python simtest.py --atmSeed $SLURM_ARRAY_TASK_ID --outdir heightPsfws --outfile outh_psfws_$SLURM_ARRAY_TASK_ID.pkl --usePsfws

python simtest.py --atmSeed $SLURM_ARRAY_TASK_ID --outdir heightRand --outfile outh_rand_$SLURM_ARRAY_TASK_ID.pkl --useRand

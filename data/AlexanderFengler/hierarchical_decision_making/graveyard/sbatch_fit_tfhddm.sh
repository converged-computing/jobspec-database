#!/bin/bash

# job name:
#SBATCH -J hddm_fitting

# priority
#SBATCH --account=carney-frankmj-condo

# email error reports
##SBATCH --mail-user=thomas_summe@brown.edu
##SBATCH --mail-type=ALL

# output file
#SBATCH --output /users/tsumme/batch_job_out/hddm_fitting_%A_%a.out

# Request runtime, memory, cores
#SBATCH --time=18:00:00
#SBATCH --mem=64G
#SBATCH -c 24
#SBATCH -N 1
##SBATCH --constraint='quadrortx'
##SBATCH --constraint='cascade'
##SBATCH -p gpu --gres=gpu:1
#SBATCH --array=1-1

##source /users/tsumme/.bashrc
##conda activate hddm
##module load cuda/10.0.130
##module load cudnn/7.6

python -u /users/tsumme/fit_hddm.py $1 $2 $3 $4

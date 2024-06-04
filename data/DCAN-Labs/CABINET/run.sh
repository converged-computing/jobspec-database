#!/bin/bash -l

#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --gres=gpu:1
#SBATCH --mem=240gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user={MAIL_USER}
#SBATCH -p v100
#SBATCH -o /path/to/logs/%A_cabinet.out
#SBATCH -e /path/to/logs/%A_cabinet.err
#SBATCH -J cabinet
#SBATCH -A {ACCOUNT}

module load singularity
module load python

singularity=`which singularity`

# conda activate /home/support/public/pytorch_1.11.0_agate

./run.py $1

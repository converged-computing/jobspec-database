#!/bin/bash

#SBATCH --job-name=tf-unet-nerve
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100GB
#SBATCH --time=10:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=jtb470@nyu.edu
module purge
unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

module load python3/intel/3.6.3

source env/bin/activate
python split_train_test.py --data /scratch/jtb470/nerve_data/train --save /scratch/jtb470/nerve_split_data --split 10


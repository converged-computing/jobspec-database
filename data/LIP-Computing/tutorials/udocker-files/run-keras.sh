#!/bin/bash
#SBATCH --job-name=run_keras
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --output=keras-%j.out
#SBATCH --error=keras-%j.err

export TUT_DIR=$HOME/udocker-tutorial
export PATH=$HOME/udocker-1.3.10/udocker:$PATH
export UDOCKER_DIR=$TUT_DIR/.udocker
module load python
cd $TUT_DIR

echo "###############################"
udocker run -v $TUT_DIR/udocker-files/tensorflow:/home/user -w /home/user tf_gpu python3 keras_2_small.py

#!/bin/bash
#SBATCH --gres gpu:1

PORT=$1
NOTEBOOK_DIR=$2
cd $NOTEBOOK_DIR

module load system
module load x11
module load stata
ml py-tensorflow/2.1.0_py36

# pip3 install ipython --upgrade --user
# pip3 install pandas --upgrade --user
# pip3 install jupyter --upgrade --user

STATATMP="/scratch/groups/maggiori/stata_tmp"
export STATATMP

~/.local/bin/jupyter lab --no-browser --port=$PORT

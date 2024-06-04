#!/bin/bash
#OAR -n "Dataset Generation"
#OAR -p gpu>0
#OAR -l /gpu=1,walltime=20:00:00
#OAR -O logs/generation.log
#OAR -E logs/generation-error.log
#OAR -t besteffort

echo " " > logs/generation-error.log
echo " " > logs/generation.log

export http_proxy=http://11.0.0.254:3142/
export https_proxy=http://11.0.0.254:3142/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

cd ~/M2-Prestel-state-from-obs-ML
NSLOTS=$(cat $OAR_NODEFILE | wc -l)

echo $NSLOTS

source ./venv/bin/activate
ls -s /scratch-local/vforiel ./M2-Prestel-state-from-obs-ML/scratch-local

python Generate_Dataset.py

exit 0
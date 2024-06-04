#!/bin/bash
#OAR -n "ML Model Training"
#OAR -p gpu>0
#OAR -l /gpu=3,walltime=20:00:00
#OAR -O logs/Train_Model_gpu.log
#OAR -E logs/Train_Model_gpu-error.log
#OAR -t besteffort

echo " " > logs/Train_Model_gpu-error.log
echo " " > logs/Train_Model_gpu.log

export http_proxy=http://11.0.0.254:3142/
export https_proxy=http://11.0.0.254:3142/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

cd ~/M2-Prestel-state-from-obs-ML
NSLOTS=$(cat $OAR_NODEFILE | wc -l)

echo $NSLOTS

source ./venv/bin/activate
python Train_Model.py

exit 0
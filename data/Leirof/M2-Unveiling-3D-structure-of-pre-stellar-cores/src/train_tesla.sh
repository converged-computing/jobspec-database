#!/bin/bash
#OAR -n "ML Model Training"
#OAR -l /nodes=1,walltime=20:00:00
#OAR -t tesla
#OAR -O logs/Train_Model_tesla.log
#OAR -E logs/Train_Model_tesla-error.log
#OAR -t besteffort

echo " " > logs/Train_Model_tesla-error.log
echo " " > logs/Train_Model_tesla.log

export http_proxy=http://11.0.0.254:3142/
export https_proxy=http://11.0.0.254:3142/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

cd ~/M2-Prestel-state-from-obs-ML
NSLOTS=$(cat $OAR_NODEFILE | wc -l)

echo $NSLOTS

source ./venv/bin/activate
python Train_Model.py

exit 0
#!/bin/bash
#OAR -n "Jupyter Session"
#OAR -p gpu>0
#OAR -l /gpu=1,walltime=10:00:00
#OAR -O logs/jupyter.log
#OAR -E logs/jupyter-error.log
#OAR -t besteffort

echo " " > logs/jupyter-error.log
echo " " > logs/jupyter.log

export http_proxy=http://11.0.0.254:3142/
export https_proxy=http://11.0.0.254:3142/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
# export CUDA_HOME=/usr/local/cuda-11.4/
# export PATH=$PATH:/usr/local/cuda-11.4/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/TensorRT-8.5.3.1/lib

cd ~/M2-Prestel-state-from-obs-ML
NSLOTS=$(cat $OAR_NODEFILE | wc -l)

echo $NSLOTS

source ~/M2-Prestel-state-from-obs-ML/venv/bin/activate

~/M2-Prestel-state-from-obs-ML/venv/bin/python ~/M2-Prestel-state-from-obs-ML/venv/bin/jupyter-lab --no-browser --port 25565 --IdentityProvider.token=c7b927e4bd9ad9f008a2491c40b8f5d4790945a935674a6e --ServerApp.allow_origin='*'

exit 0

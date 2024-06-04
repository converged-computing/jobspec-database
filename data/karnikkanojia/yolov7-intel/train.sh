#!/bin/bash
#PBS -l walltime=1:00:00
source $ONEAPI_INSTALL/setvars.sh --force > /dev/null 2>&1
source activate pytorch
pip install -r requirements.txt > /dev/null 2>&1
export OMP_NUM_THREADS=64
export OMP_SCHEDULE=STATIC
export OMP_PROC_BIND=CLOSE
echo "########## Executing the run"
echo start: $(date "+%y%m%d.%H%M%S.%3N")
numactl --cpunodebind=0 --membind=0 python detect.py --device cpu --weights yolov7.pt --conf 0.35 --source Intel-OneAPI-1/test/images --nosave
echo end: $(date "+%y%m%d.%H%M%S.%3N")
echo "########## Done with the run"


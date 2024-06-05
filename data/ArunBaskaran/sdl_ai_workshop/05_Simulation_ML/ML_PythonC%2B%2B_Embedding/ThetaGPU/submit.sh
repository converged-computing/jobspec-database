#!/bin/bash -l
#COBALT -n 1
#COBALT -t 00:10:00
#COBALT -q training-gpu
#COBALT -A SDL_Workshop

CONTAINER=/lus/theta-fs0/software/thetagpu/nvidia-containers/tensorflow2/tf2_20.08-py3.simg
SCRIPT=/home/$USER/sdl_ai_workshop/05_Simulation_ML/ML_PythonC++_Embedding/ThetaGPU/queue_submission.sh

echo "Running Cobalt Job $COBALT_JOBID."
mpirun -n 1 -npernode 1 -hostfile $COBALT_NODEFILE singularity run --nv -B /lus:/lus $CONTAINER $SCRIPT


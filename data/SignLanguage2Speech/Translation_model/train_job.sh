#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J mBART_Pretraining
### -- ask for number of cores (default: 1) --
#BSUB -n 2
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 8:00
# request 32GB of system-memory
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "select[gpu32gb]"

### -- set the email address --
#BSUB -u s200925@student.dtu.dk

### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o mBART_Pretraining-%J.out
#BSUB -e mBART_Pretraining-%J.err
# -- end of LSF options --

nvidia-smi
### Load the cuda module

module load python3/3.10.7
module load cuda/12.0
module load cudnn/v8.3.2.44-prod-cuda-11.X


source /zhome/6b/b/151617/env2/bin/activate
python3 /zhome/6b/b/151617/translation_network/main.py

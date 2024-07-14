#!/bin/bash

### -- set the job Name -- 
#BSUB -J TRAIN-SRGAN-ST[1-5]%5

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --

#BSUB -o logs/ablation_%J.out
#BSUB -e logs/ablation_%J.err
# -- end of LSF options --

### -- specify queue -- 
#BSUB -q gpua100

### -- ask for number of cores -- 
#BSUB -n 1

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"

### -- specify that we need X GB of memory per core/slot -- 
#BSUB -R "rusage[mem=5GB]"

### -- set walltime limit: hh:mm --
#BSUB -W 24:00

### -- set the email address --
#BSUB -u s204163@student.dtu.dk
### -- send notification at start --
##BSUB -B
### -- send notification at completion--
##BSUB -N

export job_index=$((LSB_JOBINDEX-1))

nvidia-smi 

source .env/bin/activate
python3 main.py

# TODO: Move results to scratch
# # Delete the sample directory afterwards
# rm -fr samples/$name
# # Move the results to scratch
# mv /zhome/c9/c/156514/SRGAN-ST/results/$name /work3/s204163/

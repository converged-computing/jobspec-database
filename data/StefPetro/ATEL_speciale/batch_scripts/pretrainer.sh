#!/bin/sh 


# Load dependencies
module load python3/3.9.14

python3 -m pip install --user -r requirements.txt

# module load matplotlib/3.4.2-numpy-1.21.1-python-3.9.6
module load pandas/1.3.1-python-3.9.14
module load cuda/11.5
module load cudnn/v8.3.0.98-prod-cuda-11.5
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set the job Name -- 
#BSUB -J pretrainer
### -- ask for number of cores (default: 1) -- 
#BSUB -n 22
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=8GB]"
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o batch_out/pretrainer.out
#BSUB -e batch_out/pretrainer.err

# here follow the commands you want to execute 
python3 pretrain.py

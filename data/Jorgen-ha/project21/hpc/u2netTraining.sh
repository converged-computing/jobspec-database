#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua10
### -- set the job Name --
#BSUB -J DL-U2
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 10:00
# request 15GB of system-memory
#BSUB -R "rusage[mem=5GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_u2net.out
#BSUB -e gpu_u2net.err
# -- end of LSF options --


# Activate venv
module load python3/3.10.12
module load cuda/12.1
source /zhome/4e/8/181483/deep-learning-project/.venv/bin/activate


# Exit if previous command failed
if [[ $? -ne 0 ]]; then
	exit 1
fi

# Run the training
python3 /zhome/4e/8/181483/deep-learning-project/src/train_u2.py

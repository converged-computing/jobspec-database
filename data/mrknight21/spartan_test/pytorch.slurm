#!/bin/bash
# SBATCH --partition=gpgpu
# Change the qos for your gpu access
# SBATCH --qos=gpgpuhpcadmin
#SBATCH --time=04:00:00
# SBATCH --gres=gpu:p100:4

module load foss/2022a 
module load PyTorch/1.12.1
python3 quickstart_tutorial.py
sleep 60
python3 tensorqs_tutorial.py 

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s


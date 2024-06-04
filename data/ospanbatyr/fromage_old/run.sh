#!/bin/bash
# 
#CompecTA (c) 2018
# 
# You should only work under the /scratch/users/<username> directory.
#
# Jupyter job submission script
#
# TODO:
#   - Set name of the job below changing "JupiterNotebook" value.
#   - Set the requested number of nodes (servers) with --nodes parameter.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter. (Total accross all nodes)
#   - Select the partition (queue) you want to run the job in:
#     - short : For jobs that have maximum run time of 120 mins. Has higher priority.
#     - mid   : For jobs that have maximum run time of 1 day..
#     - long  : For jobs that have maximum run time of 7 days. Lower priority than short.
#     - longer: For testing purposes, queue has 31 days limit but only 3 nodes.
#   - Set the required time limit for the job with --time parameter.
#     - Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input/output file names below.
#   - If you do not want mail please remove the line that has --mail-type and --mail-user. If you do want to get notification emails, set your email address.
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch jupyter_submit.sh
#
# -= Resources =-
#

#SBATCH --job-name=JupiterNotebook
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --mem=60G
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --time=1-12:00:00
#SBATCH --output=test_%J.log

# Please read before you run: http://login.kuacc.ku.edu.tr/#h.3qapvarv2g49

################################################################################
##################### !!! DO NOT EDIT BELOW THIS LINE !!! ######################
################################################################################

# Load Anaconda
echo "======================="
echo "Loading Anaconda Module..."
#module load anaconda/2.7
#module load cuda/10.2
#module load cudnn/8.1.1/cuda-10.2
module load cuda/10.2
module load cudnn/8.1.1/cuda-10.2
echo "======================="


# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

################################################################################
##################### !!! DO NOT EDIT ABOVE THIS LINE !!! ######################
################################################################################

# get tunneling info
# chexpert run
#python -u main.py --dist-backend 'nccl' --world-size 1 --rank 0 --dataset=CheXpert  --val-dataset=CheXpert --opt-version='microsoft/biogpt' --visual-model='microsoft/swin-tiny-patch4-window7-224' --exp_name='fromage_exp' --image-dir='data/'  --log-base-dir='runs/' --batch-size=64  --val-batch-size=64  --learning-rate=0.0003 --precision='fp32' --print-freq=100 --workers=2 --image-dir='/userfiles/oince22/CheXpert/chexpertchestxrays-u20210408/CheXpert-v1.0/CheXpert-v1.0' --max-len=36

# mimic run
python -u main.py --dist-backend 'nccl' --world-size 1 --rank 0 --dataset=MIMIC  --val-dataset=MIMIC --opt-version='microsoft/biogpt' --visual-model='microsoft/swin-tiny-patch4-window7-224' --exp_name='fromage_exp' --log-base-dir='runs/' --batch-size=16  --val-batch-size=16  --learning-rate=0.0003 --precision='fp32' --print-freq=100 --workers=2 --image-dir='/datasets/mimic/physionet.org/files/mimic-cxr/2.0.0/files' --max-len=100 --epochs 30

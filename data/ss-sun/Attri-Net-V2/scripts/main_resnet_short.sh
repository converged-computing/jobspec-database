#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=28:00:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti    # Partition to submit to
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=20G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=/mnt/qb/work/baumgartner/sun22/logs/hostname_%j.out  # File to which STDOUT will be written
#SBATCH --error=/mnt/qb/work/baumgartner/sun22/logs/hostname_%j.err   # File to which STDERR will be written
#SBATCH --mail-type=FAIL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=<susu.sun@uni-tuebingen.de>  # Email to which notifications will be sent


# print info about current job
scontrol show job $SLURM_JOB_ID 

# insert your commands here


# print info about current job
echo "---------- JOB INFOS ------------"
scontrol show job $SLURM_JOB_ID 
echo -e "---------------------------------\n"


# Due to a potential bug, we need to manually load our bash configurations first
source /mnt/qb/home/baumgartner/sun22/.bashrc

# cd /mnt/qb/work/baumgartner/sun22/project/tmi
cd /mnt/qb/work/baumgartner/sun22/github_projects/tmi

# Next activate the conda environment 
conda activate tt_interaction


# Run our code
echo "-------- PYTHON OUTPUT ----------"

# python3 main_resnet.py --dataset "chexpert" --epochs 20
python3 main_resnet.py --dataset "vindr_cxr" --epochs 75

echo "---------------------------------"


# Deactivate environment again
conda deactivate


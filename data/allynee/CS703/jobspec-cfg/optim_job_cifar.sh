#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ## 
#################################################

#SBATCH --nodes=1                   
#SBATCH --cpus-per-task=4           
#SBATCH --mem=16GB                  
#SBATCH --gres=gpu:1                
#SBATCH --time=24:00:00             
#SBATCH --mail-type=BEGIN,END,FAIL  
#SBATCH --output=/common/home/projectgrps/CS704/CS704G1/MetaOptNet/sbatch_logs/CIFAR_FS_MetaOptNet_RR/%u.%j.out  

# Replace below with the appropriate values from 'myinfo':
#SBATCH --partition=project        
#SBATCH --account=cs704
#SBATCH --qos=cs704qos     
#SBATCH --requeue

# Change below accordingly
#SBATCH --mail-user=allynezhang.2021@scis.smu.edu.sg,kevano.2020@business.smu.edu.sg
#SBATCH --job-name=metalearning_CIFAR_FS_MetaOptNet_RR          

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Purge the environment, load the modules we require.
module purge
module load Anaconda3/2022.05

# Create a virtual environment can be commented off if you already have a virtual environment
# Remove if needed 
# conda remove --name metalearning --all
# conda create --name metalearning python=3.6

# Do not remove this line even if you have executed conda init
eval "$(conda shell.bash hook)"

# This command assumes that you've already created the environment previously
conda activate metalearning

# Install PyTorch with the appropriate CUDA toolkit version
# conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

# Install other requirements
# pip install -r requirements.txt

# Find out which GPU you are using
srun whichgpu

# Run your command
srun python train.py --save-path "./experiments/CIFAR_FS_MetaOptNet_RR" --train-shot 5 --head Ridge --network ResNet --dataset CIFAR_FS

# Execute script with sbatch optim_job_cifar.sh

# if needed:
# cd /common/home/projectgrps/CS704/CS704G1/MetaOptNet
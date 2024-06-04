#!/bin/bash
#SBATCH --job-name="jigsaw_bot"
#SBATCH --output="testgpu.%j.%N.out"
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=95G
#SBATCH --account=aub101
#SBATCH --no-requeue
#SBATCH -t 01:00:00

DATASET="BOT"

module purge
module load gpu
module load slurm		
module load singularitypro/3.9
# by this point jigsaw must be cloned to the computer and must be currently in that folder
# load anaconda
module load anaconda3



# check if 'con_jigsaw_env' exists
# if not, create it
ENV_NAME="con_jigsaw_env"
if { conda env list | grep $ENV_NAME; } >/dev/null 2>&1; then
    echo "Conda environment for jigsaw already exists"
    echo "Activating conda environment for jigsaw"
else
    echo "Creating conda environment for jigsaw"
    conda create --name $ENV_NAME --file conda_requisites.txt
fi

conda init --all
source ~/.bash_profile
source ~/.bashrc 
conda activate $ENV_NAME

# make sure data folder exists
if [ ! -d "data" ]; then
    echo "data folder does not exist, please create the folder and place necessary dataset files."
    echo "Available at: https://github.com/jmoraga-mines/jigsawhsi#requisites"
    echo "Exiting..."
    exit 1;
fi


srun python jigsaw.py $DATASET > $DATASET.results.txt 2>&1




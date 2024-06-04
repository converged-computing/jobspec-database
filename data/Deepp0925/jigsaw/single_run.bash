#!/bin/bash
#SBATCH --job-name="single_run_task"
#SBATCH --output="testgpu.%j.%N.out"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --account=aub101
#SBATCH --no-requeue
#SBATCH -t 30:00:00

DATASET=$1
PARENT_DIR="all_results"

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

echo "Running"
# this will be the output folder for this run
OUTPUT_DIR="${PARENT_DIR}/${DATASET}/"
mkdir -p $OUTPUT_DIR
# the output file will be named 'result.txt'
OUTPUT_FILE="${OUTPUT_DIR}/${DATASET}_res.txt"
    
# get the process id
PID=$$
    
srun python jigsaw_orgi.py $DATASET > $OUTPUT_FILE 2>&1

# remove the core dump file
rm -f "core.${PID}"
# rm -f core.*

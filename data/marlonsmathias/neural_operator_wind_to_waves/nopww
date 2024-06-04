#!/bin/bash -l

# Define a job array
#SBATCH --array=1-4

# Set the slurm output file, this is where all command line output is redirected to. %j is replaced by the job id
#SBATCH --output=slurm_out_%A_%a.txt

# Define computational resources. This job requests 8 CPUs and 1 GPU in a single node.
#SBATCH -n 16 # Cores
#SBATCH -N 1 # Number of nodes
#SBATCH --gres=gpu:1

# Sepecify the partition (arandu for the DGX-A100 and devwork for the workstations)
#SBATCH -p arandu
# #SBATCH -w c4aiscw5

# Print the name of the worker node to the output file
echo "Running on"
hostname
echo "GPU $CUDA_VISIBLE_DEVICES"


# Build image
#cd /home/marlonm/neural_operator_wind_to_waves
docker build . -t nopww

# Copy files from the home folder to the output folder
mkdir /output/marlonm/NOPWW_${SLURM_ARRAY_TASK_ID}/
mkdir /output/marlonm/NOPWW_${SLURM_ARRAY_TASK_ID}/models

. ./keys.sh

# Call Docker and run the code
cd /output/marlonm/NOPWW_${SLURM_ARRAY_TASK_ID}
docker run --rm -e WANDB_API_KEY=$WANDB_API_KEY --user "$(id -u):$(id -g)" -v $(pwd)/models:/workspace/models --cpus 16 --gpus \"device=$CUDA_VISIBLE_DEVICES\" nopww wandb agent $WANDB_SWEEP

# Move the results to the home folder (Temporarily disabled)
cp /output/marlonm/NOPWW_${SLURM_ARRAY_TASK_ID}/models/* /home/marlonm/neural_operator_wind_to_waves/models/

# Clean the output folder
rm -r /output/marlonm/NOPWW_${SLURM_ARRAY_TASK_ID}
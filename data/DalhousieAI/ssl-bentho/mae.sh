#!/bin/bash
#SBATCH --time=03-12:00:00          # max walltime, hh:mm:ss
#SBATCH --nodes 1                   # Number of nodes to request
#SBATCH --gpus-per-node=a100:4      # Number of GPUs per node to request
#SBATCH --tasks-per-node=4          # Number of processes to spawn per node
#SBATCH --cpus-per-task=12          # Number of CPUs per GPU
#SBATCH --mem=498G                  # Memory per node
#SBATCH --output=logs/%x_%A-%a_%n-%t.out
                                    # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
                                    # Note: You must manually create output directory "logs" before launching job.
#SBATCH --job-name=mae
#SBATCH --account=def-ttt			# Use default account

GPUS_PER_NODE=4

# Exit if any command hits an error
set -e

#Store the time at which the script was launched
start_time="$SECONDS"

# Set and activate the virtual environment
ENVNAME=ssl_env
source ~/venvs/ssl_env/bin/activate

# Multi-threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export MASTER_ADDR=$(hostname -s)  # Store the master node’s IP address in the MASTER_ADDR environment variable.
export MAIN_HOST="$MASTER_ADDR"

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

# Get the address of an open socket
source "./slurm/get_socket.sh"

# Copy and extract data over to the node
source "./slurm/copy_and_extract_data.sh"

# Any remaining arguments will be passed through to the main script later
# (The pass-through works like *args or **kwargs in python.)
echo "EXTRA_ARGS = ${@}"

srun python ./solo_learn_train-bentho.py \
	--ssl_cfg "mae.cfg" \
	--method "mae" \
	--aug_stack_cfg "mae_aug_stack.cfg" \
	--nodes $SLURM_JOB_NUM_NODES \
	--gpus $GPUS_PER_NODE \
	--name "mae-training-full" \
	"${@}"

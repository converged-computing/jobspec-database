#!/bin/bash 
#SBATCH --partition=xeon-g6-volta 
#SBATCH --constraint=xeon-g6
#SBATCH -t 0-48:0
#SBATCH -o logs/moebert_perm_tuning/out_%A_%a.txt #redirect output to output_JOBID.txt
#SBATCH -e logs/moebert_perm_tuning/err_%A_%a.txt #redirect errors to error_JOBID.txt
#SBATCH --gres=gpu:volta:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --array=1-2


# Initialize the module command first
source /etc/profile

# Load modules
module load anaconda/2021b

# Call your script as you would from your command line
source activate MoEBERT

export TOTAL_GPUS=${SLURM_NTASKS}
export GPUS_PER_NODE=2

echo "Total number of GPUs: $TOTAL_GPUS"
echo "GPUs per node: $GPUS_PER_NODE"

if [ ! -e /proc/$(pidof nvidia-smi) ]
then
	echo "nvidia-smi does not seem to be running. exiting job"
    exit 1
fi

HF_USER_DIR="/home/gridsan/$(whoami)/.cache/huggingface"
HF_LOCAL_DIR="/state/partition1/user/$(whoami)/cache/huggingface"
mkdir -p $HF_LOCAL_DIR
rsync -a --ignore-existing $HF_USER_DIR/ ${HF_LOCAL_DIR}
export HF_HOME=${HF_LOCAL_DIR}
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_DISABLED="true"

export BACKEND="pytorch"

export HDF5_USE_FILE_LOCKING=FALSE

cd /home/gridsan/$(whoami)/MoEBERT-fork

TASK_ID=$SLURM_ARRAY_TASK_ID
EXP_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"


export DATASET=${dsets[TASK_ID%2]}
#export SEED=${seeds[TASK_ID%5]}

echo $TASK_ID

echo $EXP_ID

export output_dir="/home/gridsan/$(whoami)/MoEBERT-fork/results"

srun bash sh_scripts/experiments/launch_jobs_perm.sh squad_v2 $output_dir ${TASK_ID}

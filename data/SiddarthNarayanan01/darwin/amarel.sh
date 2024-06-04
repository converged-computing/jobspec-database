#!/bin/bash
#SBATCH --partition=gpu           # Partition (job queue)
#SBATCH --gres=gpu:3

#SBATCH --requeue                 # Return job to the queue if preempted

#SBATCH --job-name=darwin      # Assign a short name to your job

#SBATCH --nodes=1                 # Number of nodes you require

# #SBATCH --array=1-5               # Run the job 5 times

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=12   # Cores per task (>1 if multithread tasks)

#SBATCH --mem=100G                # Real memory (RAM) required

#SBATCH --time=2:00:00           # Total run time limit (HH:MM:SS)

#SBATCH --output="/scratch/%u/JOB-%j/slurm-out.out"
#SBATCH --error="/scratch/%u/JOB-%j/slurm-error.err"

module purge
module load cuda/12.1.0

cd $HOME/darwin

export OLLAMA_DEBUG=1
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED=4
export OLLAMA_HOST="0.0.0.0"

export BASE_LOG_PATH="/scratch/$USER/JOB-$SLURM_JOB_ID/"

srun --overlap -n1 ollama serve &

# Specify how many samplers the python file will spawn
srun --overlap -n1 python3 run.py &
wait

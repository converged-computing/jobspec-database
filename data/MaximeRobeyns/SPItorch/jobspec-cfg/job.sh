#!/bin/bash

#SBATCH --job-name ft_sim
#SBATCH --partition cnu
#SBATCH --ntasks 4
#SBATCH --gpus-per-task A100:1
#SBATCH --gpu-bind per_task:1
#SBATCH --cpus-per-task 8
#SBATCH --mem-per-gpu 32G
#SBATCH --time 0-10:00:00
#SBATCH --output output.log

# Usage, for local runs:
# ./job.sh path/to/file.py +hydra=arg
# or for SLURM runs:
# sbatch job.sh path/to/file.py +hydra=arg

# For 80G A100s:
# #SBATCH --nodes 2
# SBATCH --nodelist bp1-gpu[038,039]
# #SBATCH --nodelist bp1-gpu039

# 1. Print environment information
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`

export SPS_HOME=$(pwd)/deps/fsps
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export NCCL_BLOCKING_WAIT=1

# 2. If running on BluePebble cluster, load appropriate modules to setup shell
if [[ $(hostname) == bp1-* ]]; then
    module load lang/python/anaconda/3.9.7-2021.12-tensorflow.2.7.0
    eval "$(conda shell.bash hook)"
    conda activate /user/work/`whoami`/condaenvs/spivenv
fi

# 3. Get the master node address
export MASTER_ADDR=$(hostname | cut -d. -f1)

# 4. Find a free port on the master node (fall back to random assignments).
PORTS=($(comm -23 \
    <(seq 49152 65535 | sort) \
    <(/usr/sbin/ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) \
    | shuf | head -n 2))
export MASTER_PORT=${PORTS[0]:-$((32768+RANDOM%28231))}
export COMM_PORT=${PORTS[1]:-$((32768+RANDOM%28231))}

echo Using ${MASTER_ADDR}:${MASTER_PORT} as master.

# Interrupt script gracefully.
function cleanup {
    echo Stopping distributed script...
    kill $(ps aux | grep '[p]ython ./run.py' | awk '{print $2}')
    echo Script aborted.
}
trap cleanup SIGINT

# 4. Use slurm_launcher to launch SLURM job, or local launcher otherwise
if [[ -z $SLURM_JOB_ID || $SLURM_JOB_NUM_NODES == 1 ]]; then
    ./bin/local_launcher.sh $@
else
    srun bin/slurm_launcher.sh $@
fi

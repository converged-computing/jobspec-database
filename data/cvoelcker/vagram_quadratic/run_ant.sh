#!/bin/bash
#SBATCH -N 1            # number of nodes on which to run
#SBATCH --gres=gpu:1        # number of gpus
#SBATCH -p 'rtx6000,t4v1,t4v2,p100'           # partition
#SBATCH --cpus-per-task=1     # number of cpus required per task
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --time=16:00:00      # time limit
#SBATCH --mem=16GB         # minimum amount of real memory
#SBATCH --job-name=ant-vagram

source ~/.bashrc

module load cuda-11.3

conda deactivate
source ~/jax_gpu/bin/activate

export PYTHONPATH=$PYTHONPATH:/h/voelcker/Code/project_codebases/vagram_quadratic
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/h/voelcker/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

cd ~/Code/project_codebases/vagram_quadratic

echo $1
echo $2

python examples/train.py --env_name Ant-v3 --model_loss_fn $1 --max_steps 1500000 --seed $2 --model_hidden_size 128

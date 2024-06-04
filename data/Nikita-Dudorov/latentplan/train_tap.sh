#!/bin/bash

# *** bash script to run experiments on computecanada *** 

# job settings
#SBATCH --job-name=tap_hopper-medium-expert-v2
#SBATCH --account=def-martin4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --time=1-00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# define path to virtual environment
PATH_TO_ENV=env_TAP
# activate virtual environment 
source $PATH_TO_ENV/bin/activate

# load python 
module load python/3.10
# load physics simulation engine
module load mujoco
# load numpy, pandas, scipy etc
module load scipy-stack

# path variables 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nikitad/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# define run arguments
path=latentplan
name=T-1
datasets=(hopper-medium-expert-v2)  # (hopper-medium-replay-v2 hopper-medium-v2 hopper-medium-expert-v2)
device=cuda

for round in {1..1}; do
  for data in ${datasets[@]}; do
    python3 $path/scripts/train.py --dataset $data --exp_name $name-$round --tag development --seed $round --device $device
    python3 $path/scripts/trainprior.py --dataset $data --exp_name $name-$round --device $device
    for i in {1..20}; do
      python3 $path/scripts/plan.py --test_planner beam_prior --dataset $data --exp_name $name-$round --suffix $i --beam_width 64 --n_expand 4 --horizon 15 --device $device
    done
  done
done

for data in ${datasets[@]}; do
  for round in {1..1}; do
    python3 $path/plotting/read_results.py --exp_name $name-$round --dataset $data
  done
done

# all commands executed
echo "Done"

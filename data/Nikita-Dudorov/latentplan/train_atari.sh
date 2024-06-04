#!/bin/bash

# *** bash script to run experiments on computecanada *** 

# job settings
#SBATCH --job-name=tap-atari-inference
#SBATCH --account=def-martin4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128000M
#SBATCH --time=1-00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# define path to virtual environment
PATH_TO_ENV=env_TAP
# activate virtual environment 
source $PATH_TO_ENV/bin/activate

# load python 
module load python/3.10
# load compiler 
module load gcc/9.3.0
# load opencv
module load opencv/4.7.0
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
datasets=(Breakout)
device=cuda

# python3 $path/latentplan/atari/train_vae.py --device $device

for round in {1..1}; do
  for data in ${datasets[@]}; do
    # python3 $path/scripts/train.py --dataset $data --exp_name $name-$round --tag development --seed $round --device $device
    # python3 $path/scripts/trainprior.py --dataset $data --exp_name $name-$round --device $device
    for i in {1..100}; do
      python3 $path/scripts/plan.py --test_planner beam_prior --dataset $data --exp_name $name-$round --suffix $i --beam_width 64 --n_expand 4 --horizon 24 --device $device
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
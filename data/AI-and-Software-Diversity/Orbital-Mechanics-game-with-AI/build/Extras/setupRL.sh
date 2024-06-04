#!/bin/bash
#SBATCH -D /users/sbrt882/hyperion/buildrl
#SBATCH --job-name rlearning_orbit
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40GB
#SBATCH --time=48:00:00
#SBATCH -e results/%x_%j.e
#SBATCH -o results/%x_%j.o

#Enable modules command
source /opt/flight/etc/setup.sh
flight env activate gridware

#Modules required
#This is an example you need to select the modules your code needs.

module load libs/nvidia-cuda/11.2.0/bin

#Run your script.
python3 src/do_reinforcement_learning_runs.py
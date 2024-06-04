#!/bin/bash
#SBATCH --job-name=my_tensorflow_job
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=ADL_output.log
#SBATCH --error=ADL_error.log

# Load required modules or activate the virtual environment
#module load python/3.8.5

source activate tensorflow_env

# Run the TensorFlow jo
python3 -u ex1_main.py --log-interval 1 --seed 42 --epochs 20000 --model "cvit" 

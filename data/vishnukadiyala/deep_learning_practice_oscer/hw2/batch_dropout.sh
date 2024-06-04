#!/bin/bash

# Andrew H. Fagg
#
# Example with one experiment
#
# When you use this batch file:
#  Change the email address to yours! (I don't want email about your experiments!)
#  Change the chdir line to match the location of where your code is located
#
# Reasonable partitions: debug_5min, debug_30min, normal, debug_gpu, gpu
#

#
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
# memory in MB
#SBATCH --mem=1024
# The %j is translated into the job number
#SBATCH --output=results/dropout3/hw2_%j_stdout.txt
#SBATCH --error=results/dropout3/hw2_%j_stderr.txt
#SBATCH --time=00:30:00
#SBATCH --job-name=hw2_dropout3
#SBATCH --mail-user=vishnupk@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504305/deep_learning_practice/homework/hw2
#SBATCH --array=0-639
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate tf

# Change this line to start an instance of your experiment
python hw1_base_dropout.py --hidden 400 200 100 50 25 12 --activation_out linear --epochs 1000 --results_path ./results/dropout3 --exp_index $SLURM_ARRAY_TASK_ID --output_type ddtheta --predict_dim 1 --cpus_per_task $SLURM_CPUS_PER_TASK 


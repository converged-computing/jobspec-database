#!/bin/bash
# Andrew H. Fagg
#
# Example with an array of experiments
#  The --array line says that we will execute 4 experiments (numbered 0,1,2,3).
#   You can specify ranges or comma-separated lists on this line
#  For each experiment, the SLURM_ARRAY_TASK_ID will be set to the experiment number
#   In this case, this ID is used to set the name of the stdout/stderr file names
#   and is passed as an argument to the python program
#
#
# When you use this batch file:
#  Change the email address to yours! (I don't want email about your experiments)
#  Change the chdir line to match the location of where your code is located
#
# Reasonable partitions: debug_5min, debug_30min, normal
#

#SBATCH --partition=normal
#SBATCH --cpus-per-task=10
# memory in MB
#SBATCH --mem=10G
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=results/image_classification_exp%04a_stdout.txt
#SBATCH --error=results/image_classification_exp%04a_stderr.txt
#SBATCH --time=12:00:00
#SBATCH --job-name=image_classification
#SBATCH --mail-user=shyamkrishnan@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504311/hw3/
#SBATCH --array=0-4
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate tf

#shallow network
#python hw3_shyam.py @shallow.txt @exp.txt --rotation $SLURM_ARRAY_TASK_ID

#deep network
python hw3_shyam.py @deep.txt @exp.txt --rotation $SLURM_ARRAY_TASK_ID

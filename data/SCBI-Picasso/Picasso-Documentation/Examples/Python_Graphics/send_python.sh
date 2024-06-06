#!/usr/bin/env bash
# Leave only one comment symbol on selected options
# Those with two commets will be ignored:
# The name to show in queue lists for this job:
##SBATCH -J send_python.sh

# Number of desired cpus (can be in any node):
#SBATCH --ntasks=1

# Number of desired cpus (all in same node):
##SBATCH --cpus-per-task=1

# Amount of RAM needed for this job:
#SBATCH --mem=2gb

# The available nodes are: 
#     AMD nodes with 128 cores and 1800GB of usable RAM
#     AMD nodes  with 128 cores and 439GB of usable RAM
#     Intel nodes with 52  cores and 187GB of usable RAM
 
# The time the job will be running:
#SBATCH --time=0:10:00

# If you need nodes with special features you can select a constraint.
# Please, use cal by default. You will be assigned a node that satisfies your requests.
#SBATCH --constraint=cal
 
# Change "cal" by "sd" if you want to use Intel nodes and by "sr" if you want to use AMD nodes.
##SBATCH --constraint=sd
##SBATCH --constraint=sr

# To use GPU, comment out the constraint line and uncomment the following line.
##SBATCH --gres=gpu:1

# Set output and error files
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

# Leave one comment in following line to make an array job. Then N jobs will be launched. In each one SLURM_ARRAY_TASK_ID will take one value from 1 to 100
##SBATCH --array=1-100

# To load some software (you can show the list with 'module avail'):
# module load software
module load python/3.9.13

# the program to execute with its parameters:
time python python_script.py

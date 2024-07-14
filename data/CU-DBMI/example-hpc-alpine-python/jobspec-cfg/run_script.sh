#!/bin/bash

########################################################
# File description:
# An example run script for use with:
# https://github.com/CU-DBMI/example-hpc-alpine-python
#
# Expects the following sbatch exports:
#   CSV_FILEPATH:
#       A filepath for storing a CSV data file.
#
# Example Alpine command line usage:
#   $ sbatch --export=CSV_FILEPATH="/projects/$USER/data.csv" run_script.sh
#
# Referenced with modifications from:
# https://curc.readthedocs.io/en/latest/clusters/alpine/examples.html
########################################################

########################################################
# Slurm directives:
# -------------------
# Below are configurations for Slurm that specify
# which resources you'd like to use and how you'd like
# to use them on Alpine.
#
# Generally documentation on these may be found here:
# https://slurm.schedmd.com/sbatch.html
########################################################

# Indicates which Alpine-specific hardware partition you'd
# like to make use of to accomplish the work in this script.
# See: https://curc.readthedocs.io/en/latest/running-jobs/job-resources.html
#SBATCH --partition=amilan

# Provide a specific name used for identifying the job
# as it proceeds through Slurm.
#SBATCH --job-name=example-hpc-alpine-python

# Tells Slurm to gather standard output from running this
# file and write it to a specific file.
# Special variable symbols may be used here:
# %j - job ID
# %a - job array index
# %A - job array job ID
#SBATCH --output=example-hpc-alpine-python.out

# Sets a limit on the total time this work may take.
# The format below is in the form of hours:minutes:seconds.
#SBATCH --time=01:00:00

# Sets certain Alpine-specific characteristics the Slurm work
# performed. Can be one of: normal, long, mem.
# See: https://curc.readthedocs.io/en/latest/running-jobs/job-resources.html
#SBATCH --qos=normal

# Advises Slurm about the minimum nodes necessary for completing
# the work included in this script.
#SBATCH --nodes=1

# Advises Slurm about the maximum number of tasks involved
# with batch processing.
#SBATCH --ntasks=4

# Sets an email address to receive notifications from Alpine
#SBATCH --mail-user=your-email-address-here@cuanschutz.edu

# Indicate which notifications you'd like to receive from Alpine.
# This can also be set to START, END, or FAIL.
#SBATCH --mail-type=ALL

########################################################
# Module package commands:
# ------------------------
# Next, we use the module package to help load
# software which is pre-loaded on Alpine.
########################################################

# Unloads all existing modules which may have been previously loaded.
module purge

# Use module package to load Anaconda software so it may
# be used by your processes.
# Note: the numbers found after anaconda/####.## are subject
# to change depending on the versions installed by administrators.
module load anaconda/2022.10

########################################################
# Anaconda environment management:
# ---------------------------------
# Here we load the Anaconda environment to be used
# for running the Python code below.
########################################################

# Remove any existing environments that happen to have
# the same exact name.
conda env remove --name example_env -y

# Next create the environment from the yaml file.
conda env create -f environment.yml

# Then activate the environment.
conda activate example_env

########################################################
# Run a Python file (within Anaconda environment):
# ------------------------------------------------
# After loading the environment we run the Python
# code to perform the work we'd like to accomplish.
########################################################

# Run the Python file example.py which takes an
# argparse argument for use within Python processing.
#
# Note: $CSV_FILEPATH is received as an
# sbatch exported variable and sent to Python using
# the same name.
python code/example.py --CSV_FILENAME=$CSV_FILEPATH

########################################################
# Send an end signal for the logs:
# --------------------------------
# Here we add a simple echo statement to indicate
# within the logs that the work is completed.
########################################################

echo "run_script.sh work finished!"

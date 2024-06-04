#!/bin/bash
#
# CompecTA (c) 2023
#
#
# TODO:
#   - Set name of the job below changing value.
#   - Set the requested number of nodes (servers) with --nodes parameter.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter. (Total accross all nodes)
#   - Set the required time limit for the job with --time parameter.
#     - Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input/output file names below.
#   - If you do not want mail please remove the line that has --mail-type and --mail-user. If you do want to get notification emails, set your email address.
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch slurm_submit.sh
#
# -= Resources =-
#
#SBATCH --job-name=eddy_segmenter
#SBATCH --account=users
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --partition=main
#SBATCH --nodelis=nova[32]
#SBATCH --gres=gpu:1
#SBATCH --time=7-0
#SBATCH --output=./output/%j-slurm.out

INPUT_FILE=""

################################################################################
################################################################################

################################################################################
#DO NOT DELETE!
source /etc/profile.d/lmod.sh
#DO NOT DELETE!
################################################################################


# Module File
#echo "Loading Foo..."
nvidia-smi
# FILES = (../*)
pwd
eval "$(conda shell.bash hook)"
conda activate emirdlp
#echo

echo ""
echo "======================================================================================"
env
echo "======================================================================================"
echo ""

echo "======================================================================================"
# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

echo "Running Example Job...!"
echo "==============================================================================="
# Command 1 for matrix
echo "Running Python script..."
# Put Python script command below
python main.py
# Command 2 for matrix
# echo "Running G++ compiler..."
# # Put g++ compiler command below

# # Command 3 for matrix
# echo "Running compiled binary..."
# Put compiled binary command below

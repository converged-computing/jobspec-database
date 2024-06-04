#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH -J OAR-0CD
#SBATCH -o /home/sap625/logs/out/%j.out
#SBATCH -e /home/sap625/logs/err/%j.err
#SBATCH --mail-user=sap625@mst.edu
#SBATCH --mail-type=end
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2000

N_TASKS=31

# Variables, directories, etc.
PROJECT_DIR=$HOME/dev/OAR
VENV_DIR=$HOME/envs/oar
JULIA_BIN=$HOME/julia

# Date and current folder
date
ls -la

# Activate the virtual environment for the project
# source activate $VENV_DIR
source $VENV_DIR/bin/activate

# Run the full experiment from one Julia script
$JULIA_BIN $PROJECT_DIR/scripts/0_init/dist_test.jl $N_TASKS

# End with echoes
echo --- END OF CUDA CHECK ---
echo All is quiet on the western front

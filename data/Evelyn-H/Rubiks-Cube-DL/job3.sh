#!/usr/local_rwth/bin/zsh

### COMMANDS to remember:
#  - shh login:     ssh -l ppxxxxxx login18-g-1.hpc.itc.rwth-aachen.de
#  - submit job:    sbatch <script>.sh
#  - list jobs:     sacct
#  - list gpu's:    nvidia-smi



### CONFIGURATION

# memory
#SBATCH --mem-per-cpu=32G

# job name
#SBATCH --job-name=RubiksDL3

# declare the merged STDOUT/STDERR file
#commented SBATCH --output=output/output.%J.txt

# GPU
#SBATCH --gres=gpu:1

# max runing time
#SBATCH --time=48:00:00


### SCRIPT TO RUN

# change to the work directory
export PATH=$PATH:~/.local/bin
# load python
module load python/3.6.0

# enter git repo folder
cd ~/Rubiks-Cube-DL/

# run file through pipenv
# (makes sure dependencies are all there)
pipenv run python train.py --ini ini/cube3x3-zero-goal-decay-d50.ini -n run_${SLURM_JOBID}

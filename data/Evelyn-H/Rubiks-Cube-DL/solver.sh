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
#SBATCH --job-name=Solver

# declare the merged STDOUT/STDERR file
#commented SBATCH --output=output/output.%J.txt

# GPU
#SBATCH --gres=gpu:1

# max runing time
#SBATCH --time=12:00:00


### SCRIPT TO RUN

# change to the work directory
export PATH=$PATH:~/.local/bin
# load python
module load python/3.6.0

# enter git repo folder
cd ~/Rubiks-Cube-DL/

# run file through pipenv
# (makes sure dependencies are all there)
pipenv run python solver.py --env cube3x3-sticker --cuda --plot plots/ --model saves/cube3x3-sticker-sticker-d50-run_2139889/chpt_111000.dat --max-steps 10000 --samples 100 --max-depth 25

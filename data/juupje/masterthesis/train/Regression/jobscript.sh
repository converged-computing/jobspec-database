#!/bin/zsh
#SBATCH --job-name=Regression
#SBATCH --account=
### Output path for stdout and stderr
### %J is the job ID, %I is the array ID
#SBATCH --output output_%J.log
#SBATCH --error error_%J.log

### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes OR days and hours and may add or
### leave out any other parameters
#SBATCH --time=360

### Request the amount of memory you need for your job.
### You can specify this in either MB (1024M) or GB (4G).
#SBATCH --mem-per-cpu=10G

### Request a host with a Volta GPU
### If you need two GPUs, change the number accordingly
#SBATCH --gres=gpu:1

### if needed: switch to your working directory (where you saved your program)
#cd $HOME/subfolder/
#$WORKDIR will be set by manager.py
WORKDIR=$HOME/thesis/train/trainer #main script
RUNDIR= #relative dir of config and output
SCRIPTNAME=main.py
CONFIG=config.py
PLOT_SCRIPT=plot.py
source $HOME/.zshrc
cd $WORKDIR

### Load modules
#module load python
#module load cuda/11.0
#module load cudnn/8.0.5
conda activate tf

### Make sure you have 'tensorflow-gpu' installed, because using
### 'tensorflow' will lead to your program not using the requested
### GPU.
### Execute your application
python3 "$SCRIPTNAME" --outdir="$RUNDIR" --config="$CONFIG" #--plot_script="$PLOT_SCRIPT"

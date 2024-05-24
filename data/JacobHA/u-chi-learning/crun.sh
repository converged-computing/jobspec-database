#!/bin/bash
#SBATCH --job-name=u-chi
#SBATCH --time=2-23:00:00
#SBATCH --mem-per-cpu=12gb
#SBATCH --cpus-per-task=3

# Set filenames for stdout and stderr.  %j can be used for the jobid.
# see "filename patterns" section of the sbatch man page for
# additional options
#SBATCH --error=outfiles/%j.err
#SBATCH --output=outfiles/%j.out
##SBATCH --partition=AMD6276
#SBATCH --partition=Intel
##SBATCH --partition=AMD6128
# use the gpu:
##SBATCH --gres=gpu:1
##SBATCH --partition=DGXA100
##SBATCH --export=NONE
#SBATCH --array=1-10
## --begin=now+1min
echo "using scavenger"

# Prepare conda:
eval "$(conda shell.bash hook)"
conda activate /home/jacob.adamczyk001/miniconda3/envs/oblenv
export CPATH=$CPATH:$CONDA_PREFIX/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
export MUJOCO_GL="glfw"


echo "Start Run"
echo `date`

# python experiments/wandb_job.py -d cpu -env LunarLander-v2 -a u
# python experiments/rawlik_wandb_job.py -d cpu -env LunarLander-v2 -a u
# python experiments/ablation_num_nets.py -n 10 -c 10
# python experiments/local_finetuned_runs.py -c 5 -a sql -e MountainCar-v0
# python experiments/local_finetuned_runs.py -c 5 -a sql -e Acrobot-v1
# python experiments/local_finetuned_runs.py -c 5 -a sql -e CartPole-v1
# python experiments/local_finetuned_runs.py -c 5 -a sql -e LunarLander-v2

python experiments/experiment.py --do_sweep --env Acrobot-v1 --count 500

# Diagnostic/Logging Information
echo "Finish Run"
echo "end time is `date`"
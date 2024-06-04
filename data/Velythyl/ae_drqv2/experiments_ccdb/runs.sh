#!/bin/bash
#SBATCH --account=rrg-lpaull # Liam's default account: "rrg-lpaull". Liam's account: "def-lpaull". Yoshua's account: "rrg-bengioy-ad"
#SBATCH --time=36:00:00
#SBATCH --job-name=my_job_name  #Job name
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32           # CPU cores/threads
#SBATCH --gres=gpu:a100:1                  # Number of GPUs (per node) (use v:100 for beluga, v:100l for cedar, a:100 for narval).
#SBATCH --mem=64Gb                  # memory per node
#SBATCH --output=%j.out   # STDOUT
#SBATCH --mail-user=charlie.gauthier@umontreal.ca
#SBATCH --mail-type=ALL
# Load modules
module load cuda/11.2.2 cudnn/8.2.0
module load singularity/3.8
# Run training
export PYTHONPATH="${PYTHONPATH}:/home/$USER/projects/def-lpaull/$USER/ae_drqv2"
cd /home/$USER/projects/def-lpaull/$USER/deep-var-nets/src/experiments
# Use run_general_regression_hyperparameter_baselines.py to run baseline experiments.
# Datasets range from 1-15
singularity exec --nv --home /home/$USER/projects/def-lpaull/$USER/deep-var-nets/ --env WANDB_MODE="offline",WANDB_API_KEY="",WANDB_WATCH="false",HYDRA_FULL_ERROR=1 /home/$USER/projects/def-lpaull/Singularity/pred_unc.sif python run_general_regression_hyperparameter.py $SLURM_ARRAY_TASK_ID
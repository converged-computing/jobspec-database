# source this file to activate the MLMI8.1 environment
# run these steps to create environment
module purge
module load miniconda3-4.5.4-gcc-5.4.0-hivczbz
module load slurm
eval "$(conda shell.bash hook)"
conda activate /rds/project/rds-xyBFuSj0hm0/MLMI.2022-23/shared/MLMI8/vqa_env

export TRANSFORMERS_CACHE="/home/$USER/rds/hpc-work/cache"
export HF_DATASETS_CACHE="/home/$USER/rds/hpc-work/cache"
export TOKENIZER_CACHE="/home/$USER/rds/hpc-work/cache"

export WDIR="/rds/user/$USER/hpc-work/MLMI8_2022_VQA"
export BDIR="/rds/project/rds-xyBFuSj0hm0/MLMI.2022-23/shared/MLMI8/"

wandb login



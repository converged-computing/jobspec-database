#!/bin/bash 
### Comment lines start with ## or #+space 
### Slurm option lines start with #SBATCH 
### Here are the SBATCH parameters that you should always consider: 
#SBATCH --time=0-24:00:00 ## days-hours:minutes:seconds 
#SBATCH --mem 8G        ## 3000M ram (hardware ratio is < 4GB/core)  16G
#SBATCH --ntasks=1       
#SBATCH --cpus-per-task=16 
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_L0-e1
#SBATCH --output=./cluster/uzh/train_L0_v1/train_logs/train_L0-e1.out

# module load amd
module load intel

module load anaconda3
source activate sbi
module load t4

# generate dataset
# --run_simulator \
python3 -u ./src/train/train_L0.py \
--seed 0 \
--config_simulator_path './src/config/simulator_Ca_Pb_Ma.yaml' \
--config_dataset_path './src/config/dataset_Sb0_suba1_Ra0.yaml' \
--config_train_path './src/config/train_Ta1_2.yaml' \
--log_dir './src/train/logs/logs_L0_v1/log-train_L0-e1' \
--gpu \
-y > ./cluster/uzh/train_L0_v1/train_logs/train_L0-e1.log

echo 'finished simulation'

# sbatch ./cluster/dataset_gen.sh
# squeue -u $USER
# scancel 466952
# sacct -j 466952
# squeue -u $USER
# scancel --user=wehe
# squeue -u $USER
# squeue -u $USER


#!/bin/bash 
### Comment lines start with ## or #+space 
### Slurm option lines start with #SBATCH 
### Here are the SBATCH parameters that you should always consider: 
#SBATCH --time=0-24:00:00 ## days-hours:minutes:seconds 
#SBATCH --mem 4G       
#SBATCH --ntasks=1       
#SBATCH --cpus-per-task=16 
#SBATCH --job-name=train_L0_c6 ## job name 
#SBATCH --output=./cluster/uzh/train_L0_v1/train_logs/train_L0_c6.out ## standard out file 

#SBATCH --gres=gpu:T4:1

# module load amd
# module load intel

module load anaconda3
source activate sbi
module load t4
module load cuda

# generate dataset
# --run_simulator \
python3 -u ./src/train/train_L0.py \
--config_simulator_path './src/config/simulator_Ca_Pb_Ma.yaml' \
--config_dataset_path './src/config/dataset_Sa0_suba0_Rb0.yaml' \
--config_train_path './src/config/train_Ta1.yaml' \
--log_dir './src/train/logs/log-train_L0-c6' \
--gpu \
-y > ./cluster/uzh/train_L0_v1/train_logs/train_L0_c6.log

echo 'finished simulation'

# sbatch ./cluster/dataset_gen.sh
# squeue -u $USER
# scancel 466952
# sacct -j 466952
# squeue -u $USER
# scancel --user=wehe
# squeue -u $USER
# squeue -u $USER


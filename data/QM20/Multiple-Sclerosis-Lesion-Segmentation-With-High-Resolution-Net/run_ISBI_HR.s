#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --time=40:00:00
#SBATCH --mem 32000
#SBATCH --job-name=ISBI_HR_aug
#SBATCH --mail-type=END
#SBATCH --mail-user=qc690@nyu.edu
#SBATCH -p nvidia
module purge
module load all
module load cuda/10.0
source activate py362
cd /scratch/lw1474/qiming/Video/MS_Lesion_seg
python  -u train_HRNet_ISBI.py --save_dir="./results_HRNet_ISBI">>Log/log_HRNet_ISBI_aug_tmp.txt
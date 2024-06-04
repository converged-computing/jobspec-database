#!/usr/local_rwth/bin/zsh

#SBATCH --ntasks=2

#SBATCH --ntasks-per-node=2

#SBATCH --mem-per-cpu=42G

#SBATCH --cpus-per-task=2

#SBATCH --job-name=0_raw_101-120

#SBATCH --output=output.%J.txt

#SBATCH --time=10:50:00

#SBATCH --account=rwth0583

# with gpu: 

# SBATCH --gres=gpu:1

#SBATCH --mail-type=ALL

#SBATCH --mail-user=annika.stein@rwth-aachen.de

cd /home/um106329/aisafety
# module unload intelmpi; module switch intel gcc
# module load cuda/110
# module load cudnn
source ~/miniconda3/bin/activate
conda activate my-env
python3 auc_compare_raw.py "[101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120]" 0

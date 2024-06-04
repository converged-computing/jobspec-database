#!/bin/bash
#SBATCH --job-name=zoobot_COSMOS-Web
#SBATCH --output=/n03data/huertas/COSMOS-Web/jobs/bars_zoobot.out
#SBATCH --error=/n03data/huertas/COSMOS-Web/jobs/bars_zoobot.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00


module load intelpython/3-2024.1.0
module load gcc/13.2.0

cd /n03data/huertas/python/ceers/make_stamps
python make_stamps.py
#python match_catalogs_candide.py
#python train_on_gz_ceers_tree.py
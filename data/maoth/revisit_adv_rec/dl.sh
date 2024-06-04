#!/bin/bash
#SBATCH -A r00066
#SBATCH -J train
#SBATCH -p gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xxx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --gpus-per-node v100:2
#SBATCH --time=48:00:00
#SBATCH --mem=300G

module load cudatoolkit/10.2
module load anaconda/python3.8/2020.07
source activate round10

python -u generate_attack.py \
-dataset ml-1m \
-att_type DQN \
-pop upper \
-ratio 1 \
-unroll 0 \
-tag None


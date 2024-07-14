#!/bin/bash
#SBATCH -p psych_gpu  # psych_day,psych_gpu,psych_scavenge,psych_week， psych_scavenge
#SBATCH --job-name=LA
#SBATCH --ntasks=1 --nodes=1
#SBATCH --time=6:00:00
#SBATCH --output=logs/%J.out  # %A_%a.out
#SBATCH --mem=100g
#SBATCH --gpus 1

set -e
nvidia-smi
cd /gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch/
. /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh
conda activate py36
python --version
python -u /gpfs/milgram/pi/turk-browne/projects/sandbox/sandbox/docker/hello.py

cd /gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch
CUDA_VISIBLE_DEVICES=0 python -u ./scripts/instance.py  "${1}"

nvidia-smi

echo "done"


# cd /gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch
# sbatch ./scripts/instance.sh  # 25547786 25547831 25547832 25547836 25547837 25547933 25547938 25547941 25547946
# conda activate py36 ; python -u ./scripts/instance.py  "/gpfs/milgram/project/turk-browne/projects/LocalAggregation-Pytorch/config/imagenet_la.json"

#!/usr/bin/env bash
#SBATCH -p bme_gpu
#SBATCH --job-name=DINO_COCO
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH -t 5-00:00:00

set -x

CONFIG=$1
PY_ARGS=${@:2}
source activate det
nvidia-smi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -u tools/train.py ${CONFIG} --launcher="slurm" ${PY_ARGS}
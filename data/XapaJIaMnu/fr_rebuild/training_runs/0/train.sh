#!/usr/bin/env bash

#SBATCH --qos epsrc
#SBATCH -t 12:00:00
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
#SBATCH --gres gpu:4
#SBATCH --cpus-per-gpu 36
#SBATCH --mem-per-gpu 122G

#SBATCH --array=0-16
#SBATCH --signal=TERM@120

BASE="/bask/projects/x/xngs6460-languages/gnail/enfr"

# Env
. ${BASE}/software/env.sh
MARIAN="${BASE}/software/source/marian-dev/build"

compute="--devices $(nvidia-smi --query-gpu=index --format=csv,nounits,noheader | tr '\n' ' ')"

# mkdir -p /scratch-global/slurm-jobs/$USER

$MARIAN/marian -c teacher.yml \
  --train-sets ${BASE}/data/dedup_fren/concat/combined.tsv.gz \
  --tempdir /tmp \
  --seed 1111 \
  ${compute} "${@}"

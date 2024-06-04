#!/bin/bash

#SBATCH --job-name=getdata

#SBATCH --qos=qos_gpu-t3

#SBATCH --output=./logfiles/logfile_wmt_lm.out

#SBATCH --error=./logfiles/logfile_wmt_lm.err

#SBATCH --time=09:00:00

#SBATCH --ntasks=1

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=40

#SBATCH --hint=nomultithread


module purge
module load anaconda-py3/2019.03
conda activate retrocode
set -x
nvidia-smi
# This will create a config file on your server


bash dataset/nl/seq2seq/en2fr/prepare-wmt14en2fr.sh
bash dataset/nl/seq2seq/en2de/prepare-wmt14en2de.sh

bash dataset/nl/lm/en2fr/prepare-wmt14en2fr.sh
bash dataset/nl/lm/en2de/prepare-wmt14en2de.sh
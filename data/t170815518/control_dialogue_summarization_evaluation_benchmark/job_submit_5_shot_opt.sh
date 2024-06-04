#!/bin/bash
#PBS -N five_shot_opt-iml
#PBS -l ngpus=2,mem=100gb
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -o out-run-5_.txt
#PBS -P 13003087
#PBS -q normal

export CUDA_LAUNCH_BLOCKING=1
echo "Loading modules"
module load pytorch/1.11.0-py3-gpu
module load git/2.39.2
cd ${PBS_O_WORKDIR} || exit
echo "${PBS_O_WORKDIR}"
echo "pip install packages"
pip install git+https://github.com/huggingface/transformers
pip install -r requirements.txt
echo "Running script"
python direct_prompt.py --model facebook/opt-iml-30b -k 5 --demonstration_file /home/users/ntu/ytang021/nlp_research_intern/demonstration_pairs/demonstration_samsum_k5.pickle --dataset samsum

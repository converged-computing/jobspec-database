#!/bin/bash
#PBS -q windfall
#PBS -l select=1:ncpus=28:mem=168gb:pcmem=6gb:ngpus=1:os7=True
#PBS -W group_list=msurdeanu
#PBS -l walltime=48:00:00
#PBS -j oe


cd /xdisk/msurdeanu/mithunpaul/
module load cuda90/neuralnet/7/7.3.1.20
module load python/3.6/3.6.5

#uncomment this if you don't want to reinstall venv- usually you just have to do this only once ever
rm -rf sandeep_bert_venv
mkdir sandeep_bert_venv
python3 -m venv sandeep_bert_venv

#this is the only line you need if you already have a virtual_env set up
source sandeep_bert_venv/bin/activate

cd /xdisk/msurdeanu/mithunpaul/bert_tensorflow

pip install --upgrade pip
#pip install torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt




#####my code part
export PYTHONPATH="/home/u11/mithunpaul/sandeep_bert/"





bash run_sandeep_code.sh
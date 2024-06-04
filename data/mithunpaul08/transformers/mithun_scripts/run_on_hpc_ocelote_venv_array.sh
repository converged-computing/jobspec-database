#!/usr/bin/env bash
# Your job will use 1 node, 28 cores, and 168gb of memory total.
#PBS -q windfall
#PBS -l select=1:ncpus=28:mem=168gb:pcmem=6gb:ngpus=1:os7=True
### Specify a name for the job
#PBS -N 8glbunfnf

### Specify the group name
#PBS -W group_list=msurdeanu
### Used if job requires partial node only
#PBS -l place=pack:shared
### Walltime is how long your job will run
#PBS -l walltime=55:00:00
### Joins standard error and standd out
#PBS -j oe




cd /home/u11/mithunpaul/

module load cuda90/neuralnet/7/7.3.1.20
module load python/3.6/3.6.5

#uncomment this if you don't want to reinstall venv- usually you just have to do this only once ever
#rm -rf my_virtual_env
#mkdir my_virtual_env
#python3 -m venv my_virtual_env
#####



#this is the only line you need if you already have a virtual_env set up
source my_virtual_env/bin/activate

export PYTHONPATH="/home/u11/mithunpaul/xdisk/fnc2fever_gl_bert_base_uncased_rs8/code/src"

export CUDA_VISIBLE_DEVICES=0

pip install --upgrade pip
pip install -U spacy
pip install stop-words
python -m spacy download en_core_web_sm

   


cd /home/u11/mithunpaul/xdisk/lexStandAlone_fever2fnc_distilbert_base_uncased_rs3082/code/examples


pip install -r requirements.txt
pip install transformers
pip install wget
pip install stop-words  --no-cache-dir


cd /home/u11/mithunpaul/xdisk/fnc2fever_gl_bert_base_uncased_rs8/code/mithun_scripts

bash run_all.sh --epochs_to_run 55 --machine_to_run_on hpc --use_toy_data false --download_fresh_data true #options include [laptop, hpc,clara]





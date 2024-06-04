#!/bin/bash
#SBATCH -J Simple_Language_Model
#SBATCH -N 1
#SBATCH --mem=16384
#SBATCH --account=HSTR_EinfacheSprache
#SBATCH -t 20:00:00
#SBATCH -o SLM1e-3_V1_EP20_PT3-%j.out
#SBATCH -e SLM1e-3_V1_EP20_PT3-%j.err
#SBATCH --gres=gpu:V100:1 # select a host with a Volta GPU
#SBATCH --mail-type=END


rhrk-singularity tensorflow_22.03-tf2-py3.simg python3 simple_language_model_test.py

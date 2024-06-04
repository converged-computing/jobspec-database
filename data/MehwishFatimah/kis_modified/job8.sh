#!/bin/bash
#SBATCH --job-name=sim8a
#SBATCH --output=/hits/basement/nlp/fatimamh/outputs/hipo/exp08/out-%j 
#SBATCH --error=/hits/basement/nlp/fatimamh/outputs/hipo/exp08/err-%j
#SBATCH --time=14-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=pascal-deep.p

module load CUDA/10.0.130

. /home/fatimamh/anaconda3/etc/profile.d/conda.sh
conda activate kis
python /hits/basement/nlp/fatimamh/codes/keep_it_simple/run_keep_it_simple.py -d /hits/basement/nlp/fatimamh/outputs/hipo/exp08/wiki_mono_test_no_sections-pacsum_bert-cos-order-add_f=0.0_b=1.0_s=1.0 
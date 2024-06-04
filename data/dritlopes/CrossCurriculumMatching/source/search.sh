#!/bin/bash

# https://userinfo.surfsara.nl/systems/lisa/user-guide/creating-and-running-jobs

#Set job requirements
#SBATCH -N 1
#SBATCH -t 100:00:00
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=adrielli.drica@gmail.com


#Loading modules
module load 2020
module load Anaconda3/2020.02
module load Python/3.8.2-GCCcore-9.3.0
python --version


# install environment
# pip install -r $HOME/query_matching/query_matching/requirements.txt
# conda env create -f $HOME/query_matching/query_matching/venv.yml
#conda create --name venv2020 --file $HOME/query_matching/query_matching/requirements.txt
source /sw/arch/Debian10/EB_production/2020/software/Anaconda3/2020.02/etc/profile.d/conda.sh
conda activate venv2020
# conda install --file $HOME/query_matching/query_matching/requirements.txt
conda install -c conda-forge sentence-transformers
conda install scikit-learn
conda install pandas
conda install pytorch
conda install nltk
conda install matplotlib
conda install seaborn
conda install openpyxl
conda install -c conda-forge fasttext
conda install scipy
conda install -c conda-forge lightgbm

##Copy input data to scratch and create output directory
#cp -r $HOME/query_matching/query_matching/data "$TMPDIR"
#mkdir "$TMPDIR"/output_dir
#
#

##Run program
#python $HOME/query_matching/query_matching/source/train_classifier.py "TMPDIR"/query_matching/query_matching/data "$TMPDIR"/output_dir
#
#
##Copy output data from scratch to home
#cp -r "$TMPDIR"/output_dir $HOME/query_matching/query_matching

python $HOME/query_matching/query_matching/source/main.py $HOME/query_matching/query_matching/data $HOME/query_matching/query_matching/models $HOME/query_matching/query_matching/eval $HOME/query_matching/query_matching/results
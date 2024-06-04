#!/bin/bash -l
#PBS -l nodes=1:ppn=8:gtx1080,walltime=24:00:00
#PBS -N pytorch-style

module load python/3.7-anaconda

cd ~/Style_Classification_in_Posters_IT-Project/Code/Scripts
python training.py --epochs 100 --batch_size 40

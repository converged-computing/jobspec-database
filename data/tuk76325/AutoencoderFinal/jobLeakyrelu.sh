#!/bin/sh
#PBS -l walltime=1:00:00
#PBS -N mytestjob
#PBS -q large

module load cuda

export PATH=/home/tuk76325/anaconda3/bin:$PATH

source activate pytorch
cd /home/tuk76325/work/PythonProjects/MinaAutoEncoder
python main.py 80 leakyrelu
#!/bin/bash

#PBS -N MTD
#PBS -l nodes=1:ppn=32
#PBS -l mem=128gb

module load python/3.6.0
#module load cuda-toolkit/6.5
#module load gcc/5.4.0
#module load cmake/3.6.3
#module load autoconf/2.69

cd MovingTargetDefenceSimulator
source venv/bin/activate

pip install -r requirements.txt

python Trainer.py

deactivate
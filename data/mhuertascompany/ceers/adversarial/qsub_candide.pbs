#!/bin/sh
#PBS -S /bin/sh
#PBS -N morpho_COSMOS
#PBS -o /n03data/huertas/COSMOS-Web/jobs/qsub_adversarial.out 
#PBS -j oe 
#PBS -l nodes=n03:ppn=4,walltime=50:00:00

module () {
  eval $(/usr/bin/modulecmd bash $*)
}
module load tensorflow/2.9
#module load cuda/11.7

# Set CUDA_VISIBLE_DEVICES to use GPU device 1
export CUDA_VISIBLE_DEVICES=0

cd /home/huertas/python/ceers/adversarial
python adversarial_training_COSMOS-Web.py
#python adversarial_training_combined.py
exit 0  
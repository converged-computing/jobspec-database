#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --ntasks-per-node=1
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --nodelist=compute-0-[9]
#SBATCH --mail-type=END,FAIL             # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=stirumal@andrew.cmu.edu     # Where to send mail

set -x
set -u
set -e
module load singularity
module load cuda-80

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 singularity exec --nv /home/stirumal/singularity/hw4p2.sif python train.py --lr 3e-4 -wd 1e-6 -bs 64 -e 100 -nl 4 -nld 3 -ed 256 -dd 256 -ebd 128 -kvs 256 -sim 0 -dp /scratch/sashank/datasets/hw4p2_student_data/hw4p2_student_data -rp /scratch/sashank/runs/hw4p2/r1 -w 1 -wu 12 -drp 0.3
#!/bin/bash
#SBATCH --job-name=cdropfin7    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=2G         # memory per cpu-core (4G is default)
#SBATCH --time=0-01:00:00        # total run time limit (days-HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --mail-user=aalag@princeton.edu
#SBATCH --output=slurm-%j-%x.out

module purge
module load anaconda3/2022.5
conda activate audit
wandb offline

epoch=50
# I=$1
I=0.7

## Train the base + calibration model and run audit
experiment="COVIDx_dropout$I"
dataset="COVIDx"

# Train the base model
# python train_model.py --mode base --dataset $dataset --batch_size 64 --epoch $epoch --train_size 10000 --dropout $I --expt $experiment

# train the calibration model
# for k in 0 10 20 30 40 50
# do
#     echo k $k
#     python train_model.py --mode cal --dataset $dataset --batch_size 64 --epoch $epoch --train_size 10000 --k $k --cal_data $dataset --dropout $I --expt $experiment
# done

# python train_model.py --mode cal --dataset $dataset --batch_size 64 --epoch $epoch --train_size 10000 --k 50 --cal_data $dataset --dropout $I --expt $experiment

# # evaluate the audit based on memguard flags
for k in 0 10 20 30 40 50
do
    for fold in 0 1 2 3 4 5 6
    do
        echo k $k fold $fold
        python run_audit.py --k $k --fold $fold --audit EMA --epoch $epoch --cal_data $dataset --dataset $dataset --cal_size 10000 --expt $experiment --dropout $I
    done
done

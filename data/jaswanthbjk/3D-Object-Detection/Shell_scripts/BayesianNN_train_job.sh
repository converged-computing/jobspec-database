#!/bin/bash
#SBATCH --job-name=Bayes-fpointnet
#SBATCH --partition=gpu
#SBATCH --mem=64G            # memory per node in MB (different units with$
#SBATCH --ntasks-per-node=32    # number of cores
#SBATCH --time=0-72:00:00           # HH-MM-SS
#SBATCH --output=/home/jbandl2s/train.%j.out # filename for STDOUT (%N: nodename, %j: j$
#SBATCH --error=/home/jbandl2s/train.%j.err  # filename for STDERR
#SBATCH --gres=gpu:0


# load cuda
module load cuda

# activate environment
#source ~/anaconda3/bin/activate ~/anaconda3/envs/RnD

# locate to your root directory 
cd /home/jbandl2s/Reference

# run the script
DATA_FILE="/scratch/jbandl2s/Lyft_dataset/artifacts/augmented_train_data"
MODEL_LOG_DIR="./log_v1_test/"
RESTORE_MODEL_PATH="./log_v1_test/model.ckpt"

python train_v3.py --gpu 0 --model Bayes_F_pointnet --log_dir "./log_v1_test/" --max_epoch 200 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --data_dir "/scratch/jbandl2s/Lyft_dataset/artifacts/frustums_train" --is_bayesian True




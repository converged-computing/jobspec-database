#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=p100_4
#SBATCH --mem=100GB
#SBATCH --job-name=PTB_pipe
#SBATCH --mail-type=END
#SBATCH --mail-user=gz612@nyu.edu
#SBATCH --output=logs/ptb_pipe.out


# configuration type to select in config.yml
# if not defined, will use the default configuration
CONFIG='small_nets'


# load required modules
module purge
module load tensorflow/python3.5/1.2.1 cuda/8.0.44

cd hierarchical-rnn
# train on Text8 dataset
python3 -u char_class.py --config $CONFIG > logs/char_class.log
# test on Penn Treebank
python3 -u ptb_test.py --config $CONFIG > logs/ptb_test.log

# backup generated tensorflow models
TIMESTAMP=$(date +%Y%m%d%H%M%S)
mkdir -p ../backup/$TIMESTAMP
cp logs/char_class.log checkpoint text8.[dim]* ../backup/$TIMESTAMP/

# evaluate boundary indicators by PTB as benchmark
cd ../treebank
python3 evaluate.py > logs/ptb_eval.log

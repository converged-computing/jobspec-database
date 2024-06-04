#!/bin/bash
#SBATCH --array=1-100
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=1-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_analytic_Gridsearch
#SBATCH --output=%x-%j.out
source $HOME/environments/carrim/bin/activate
python $ARIM/rim_gridsearch.py\
  --n_models=100\
  --strategy=uniform\
  --steps 4 8 12\
  --adam 1\
  --units 16 32 64\
  --cnn_architecture custom perreault_levasseur2016 resnet50 resnet101 inceptionV3\
  --cnn_levels 3 4\
  --cnn_layer_per_level 1 2\
  --cnn_input_kernel_size 7 11\
  --cnn_filters 16 32\
  --cnn_activation relu tanh swish\
  --activation tanh elu leaky_relu\
  --batch_size 32\
  --total_items 20000\
  --epochs 1000\
  --optimizer adamax\
  --initial_learning_rate 1e-4 1e-5 1e-6\
  --decay_rate 1 0.9 0.8\
  --decay_steps 10000\
  --max_time 23.5\
  --checkpoints=10\
  --max_to_keep=1\
  --model_dir=$ARIM/models/\
  --logdir=$ARIM/logsA/\
  --logname_prefixe=RIMA_g1\
  --seed 42

#!/bin/bash

# Please adjust these settings according to your needs.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
# Purge modules
module purge

# Load Singularity container
singularity exec --nv \
  --overlay /scratch/wz1492/overlay-25GB-500K.ext3:ro \
  /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
  /bin/bash -c "source /scratch/wz1492/env.sh;"

# Define hyperparameter combinations
models=("resnet50" "squeezenet" "mobilenetv2" "efficientnetb0" "inceptionv3"  "resnet101"  "resnet152" "visiontransformer" "swintransformer")
batch_sizes=(32)
learning_rates=(0.001 0.0001)
num_epochs=(60)

# Other parameters
data_dir="data/train"
num_classes=12
test_size=0.2
val_size=0.2

# Iterate over hyperparameter combinations
for model in "${models[@]}"; do
  for batch_size in "${batch_sizes[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
      for num_epoch in "${num_epochs[@]}"; do
        run_name="${model}_bs${batch_size}_lr${learning_rate}_epochs${num_epoch}"
        echo "Running: $run_name"

        python main.py \
          --model "$model" \
          --data_dir "$data_dir" \
          --num_classes "$num_classes" \
          --batch_size "$batch_size" \
          --learning_rate "$learning_rate" \
          --num_epochs "$num_epoch" \
          --test_size "$test_size" \
          --val_size "$val_size"
      done
    done
  done
done
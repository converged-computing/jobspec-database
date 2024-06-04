#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=TrainResnet18
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=job-outputs/resnet_18_%A.out

module purge
module load 2022
module load Anaconda3/2022.05

# activate the environment
source activate dl2023

# Go to directory
cd $HOME/deep-learning-assignments/assignment_2/part1/

# Run the script
echo "Running experiment without the augmentation"
srun python train.py --checkpoint_name ./save/models/fine-tuned-resnet18 --data_dir $TMPDIR/data/

# Run the script
augmentations=("flip" "resize")

for augmentation in "${augmentations[@]}"; do
    echo "Running experiment with the augmentation: $augmentation"
    srun python train.py --checkpoint_name ./save/models/fine-tuned-resnet18-$augmentation --data_dir $TMPDIR/data/ --augmentation_name $augmentation
done

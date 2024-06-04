#!/bin/bash

#SBATCH --job-name=ewc_eval
#SBATCH --mail-user=adivasu@umich.edu
#SBATCH --mail-type=END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=30g
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --account=eecs602w24_class
#SBATCH --partition=spgpu
#SBATCH --output=./jobs/%u/%x-%j.log

# module load cuda
# source ../cleanrl-env/bin/activate

cd GroundingDINO
python setup.py develop
# pip install -e .
cd ..

# Epoch -1 (prior to any training)
WEIGHTS="/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/dinosaur/model_weights/gdinot-1.8m-odvg.pth"
python eval.py --real --spatial --weights=$WEIGHTS # refcoco spatial
python eval.py --real --weights=$WEIGHTS # refcoco nonspatial
python eval.py --weights=$WEIGHTS # synthetic validation

LOCATION="/scratch/eecs545w24_class_root/eecs545w24_class/shared_data/dinosaur/model_checkpoints/ewc_mixed_checkpoints"

for i in 0 1 2 3 4 5 6;
do
    WEIGHTS="$LOCATION/checkpoint000$i.pth"
    python eval.py --real --spatial --weights=$WEIGHTS # refcoco spatial
    python eval.py --real --weights=$WEIGHTS # refcoco nonspatial
    python eval.py --weights=$WEIGHTS # synthetic validation
done